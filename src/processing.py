from pathlib import Path

import numpy as np
import pandas as pd
import phik  # Регистрирует метод DataFrame.phik_matrix.
from scipy import stats


class DataProcessor:
    ORDER_LABELS = ['1 заказ', '2 заказа', '3–4 заказа', '5+ заказов']
    FIRST_FEATURES = [
        'first_device', 'first_region', 'first_event_type', 'first_weekday',
        'first_revenue_rub', 'first_tickets',
    ]

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def _read_table(self, name, columns, dtype=None):
        patterns = (f'{name}.csv', f'{name}_*.csv', f'{name}.csv.gz', f'{name}_*.csv.gz')
        files = sorted(path for pattern in patterns for path in self.data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f'В {self.data_dir} нет CSV для таблицы {name}.')
        if len(files) > 1:
            filenames = ', '.join(path.name for path in files)
            raise ValueError(f'Для таблицы {name} найдено несколько выгрузок — {filenames}.')
        return pd.read_csv(
            files[0], usecols=columns, dtype=dtype, encoding='utf-8-sig',
            float_precision='round_trip',
        )

    def load_orders(self):
        purchases = self._read_table(
            'purchases',
            [
                'user_id', 'device_type_canonical', 'order_id', 'created_dt_msk',
                'created_ts_msk', 'currency_code', 'revenue', 'tickets_count',
                'event_id', 'service_name',
            ],
            dtype={'user_id': 'string'},
        )
        events = self._read_table(
            'events', ['event_id', 'event_name_code', 'event_type_main', 'city_id'],
        )
        cities = self._read_table('city', ['city_id', 'city_name', 'region_id'])
        regions = self._read_table('regions', ['region_id', 'region_name'])

        purchases = purchases.loc[purchases['device_type_canonical'].isin(['mobile', 'desktop'])]
        event_type = events['event_type_main']
        events = events.loc[event_type.notna() & event_type.ne('фильм')]

        return (
            purchases.merge(events, on='event_id', how='inner', validate='many_to_one')
            .merge(cities, on='city_id', how='left', validate='many_to_one')
            .merge(regions, on='region_id', how='left', validate='many_to_one')
            .drop(columns=['city_id', 'region_id'])
            .rename(columns={
                'created_dt_msk': 'order_dt',
                'created_ts_msk': 'order_ts',
                'event_name_code': 'event_name',
            })
        )

    def load_rates(self):
        rates = pd.read_csv(self.data_dir / 'final_tickets_tenge_df.csv')
        rates['order_dt'] = pd.to_datetime(rates['data']).dt.normalize()
        valid_rates = rates['curs'].gt(0) & np.isfinite(rates['curs'])
        if (
            rates['order_dt'].isna().any()
            or rates['order_dt'].duplicated().any()
            or not (rates['cdx'].eq('kzt') & rates['nominal'].eq(100) & valid_rates).all()
        ):
            raise ValueError('Ожидается один положительный курс за 100 KZT на дату.')
        return rates.set_index('order_dt')['curs']

    def preprocess(self, raw, rates):
        orders = raw.copy()
        if orders.empty:
            raise ValueError('После отбора и объединения CSV не осталось заказов.')

        orders['order_dt'] = pd.to_datetime(orders['order_dt']).dt.normalize()
        orders['order_ts'] = pd.to_datetime(orders['order_ts'], format='mixed')
        for col in ['order_id', 'event_id', 'tickets_count']:
            orders[col] = pd.to_numeric(orders[col], downcast='integer')
        orders['revenue'] = pd.to_numeric(orders['revenue'])
        required = ['user_id', 'order_id', 'event_id', 'order_ts', 'order_dt']
        if (
            orders[required].isna().any().any()
            or orders['order_id'].duplicated().any()
            or not orders['order_ts'].dt.normalize().eq(orders['order_dt']).all()
            or not np.isfinite(orders['revenue']).all()
            or not (orders['tickets_count'].ge(1) & orders['tickets_count'].mod(1).eq(0)).all()
        ):
            raise ValueError('Проверь ID, даты, выручку и количество билетов в выгрузке покупок.')

        kzt = orders['currency_code'].eq('kzt')
        if not orders['currency_code'].isin(['rub', 'kzt']).all():
            raise ValueError('Выгрузка содержит валюты кроме RUB и KZT.')
        kzt_rates = orders.loc[kzt, 'order_dt'].map(rates)
        if kzt_rates.isna().any():
            raise ValueError('Для некоторых заказов в тенге отсутствует курс на дату покупки.')
        orders['revenue_rub'] = orders['revenue'].astype(float)
        orders.loc[kzt, 'revenue_rub'] = orders.loc[kzt, 'revenue'] * kzt_rates / 100

        orders = orders.sort_values(['user_id', 'order_ts', 'order_id'])
        duplicate_key = [
            'user_id', 'order_ts', 'device_type_canonical', 'service_name',
            'event_id', 'revenue', 'currency_code', 'tickets_count',
        ]
        orders = orders.drop_duplicates(duplicate_key).copy()
        removed_duplicates = len(raw) - len(orders)

        for col in [
            'device_type_canonical', 'currency_code', 'service_name',
            'event_type_main', 'region_name', 'city_name',
        ]:
            orders[col] = (
                orders[col].astype('string').str.strip().replace('', 'неизвестно')
                .fillna('неизвестно').astype('category')
            )

        revenue_p99 = orders['revenue_rub'].quantile(.99)
        cleaned = orders.loc[orders['revenue_rub'].le(revenue_p99)].copy()
        cleaned['days_since_prev'] = (
            cleaned.groupby('user_id')['order_dt'].diff().dt.days.astype('Int16')
        )
        cleaned['revenue_per_ticket_rub'] = cleaned['revenue_rub'] / cleaned['tickets_count']
        return cleaned, {
            'duplicates': removed_duplicates,
            'revenue_p99': revenue_p99,
            'expensive_orders': len(orders) - len(cleaned),
        }

    def create_user_profiles(self, orders, observation_end):
        # Предобработка сортирует заказы по времени и ID.
        users = orders.groupby('user_id', observed=True).agg(
            first_order_dt=('order_dt', 'min'),
            last_order_dt=('order_dt', 'max'),
            first_order_ts=('order_ts', 'first'),
            first_device=('device_type_canonical', 'first'),
            first_region=('region_name', 'first'),
            first_event_type=('event_type_main', 'first'),
            first_revenue_rub=('revenue_rub', 'first'),
            first_tickets=('tickets_count', 'first'),
            total_orders=('order_id', 'size'),
            avg_revenue_rub=('revenue_rub', 'mean'),
            total_revenue_rub=('revenue_rub', 'sum'),
            total_tickets=('tickets_count', 'sum'),
            avg_tickets=('tickets_count', 'mean'),
            avg_days_between=('days_since_prev', 'mean'),
        ).reset_index()
        users['is_two'] = users['total_orders'].ge(2)
        users['avg_revenue_per_ticket_rub'] = users['total_revenue_rub'] / users['total_tickets']
        users['first_weekday'] = users['first_order_dt'].dt.dayofweek
        users['first_month'] = users['first_order_dt'].dt.strftime('%Y-%m')
        users['observed_span_days'] = (users['last_order_dt'] - users['first_order_dt']).dt.days
        users['observation_days'] = (observation_end - users['first_order_dt']).dt.days

        nonnegative = users.loc[users['avg_revenue_rub'].ge(0)]
        orders_p99 = nonnegative['total_orders'].quantile(.99)
        filtered = nonnegative.loc[nonnegative['total_orders'].le(orders_p99)].copy()
        if filtered.empty:
            raise ValueError('После фильтрации не осталось пользовательских профилей.')
        filtered['order_segment'] = pd.cut(
            filtered['total_orders'], [0, 1, 2, 4, np.inf], labels=self.ORDER_LABELS,
        )
        return filtered, {
            'negative_profiles': len(users) - len(nonnegative),
            'active_profiles': len(nonnegative) - len(filtered),
        }

    def calculate_retention(self, users):
        baseline = users['is_two'].mean()
        segments = {}
        for col in [
            'first_event_type', 'first_device', 'first_region',
            'first_weekday', 'first_month', 'first_tickets',
        ]:
            table = users.groupby(col, observed=True).agg(
                users=('user_id', 'size'),
                returned=('is_two', 'sum'),
                return_rate=('is_two', 'mean'),
            ).sort_values('users', ascending=False, kind='stable')
            p, n = table['return_rate'], table['users']
            z = 1.96
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
            table['ci_low'] = (center - half).clip(0, 1)
            table['ci_high'] = (center + half).clip(0, 1)
            table['difference_pp'] = 100 * (p - baseline)
            if col == 'first_month':
                table['median_observation_days'] = (
                    users.groupby(col, observed=True)['observation_days'].median()
                )
            segments[col] = table
        return segments

    def summarize_revenue(self, users):
        return users.groupby('is_two').agg(
            users=('user_id', 'size'),
            mean_order=('avg_revenue_rub', 'mean'),
            median_order=('avg_revenue_rub', 'median'),
            q25_order=('avg_revenue_rub', lambda revenue: revenue.quantile(.25)),
            q75_order=('avg_revenue_rub', lambda revenue: revenue.quantile(.75)),
            median_ticket=('avg_revenue_per_ticket_rub', 'median'),
            avg_tickets=('avg_tickets', 'mean'),
        )

    def analyze_weekdays(self, users):
        contingency = pd.crosstab(users['first_weekday'], users['is_two'])
        if min(contingency.shape) < 2:
            return None
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        # В таблице два столбца, поэтому min(r - 1, c - 1) = 1.
        cramers_v = np.sqrt(chi2 / len(users))
        return p_value, cramers_v

    def test_hypotheses(self, users, segments):
        top_regions = segments['first_region'].head(3).index
        in_top = users['first_region'].isin(top_regions)
        event = users['first_event_type']
        comparisons = [
            ('Спорт > концерты', event.eq('спорт'), event.eq('концерты')),
            ('Топ-3 регионов > остальные', in_top, ~in_top),
        ]
        results = []
        for name, mask_a, mask_b in comparisons:
            a = users.loc[mask_a, 'is_two']
            b = users.loc[mask_b, 'is_two']
            p_value = np.nan
            if len(a) and len(b):
                _, p_value = stats.fisher_exact(
                    [[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]],
                    alternative='greater',
                )
            results.append({
                'hypothesis': name, 'n_a': len(a), 'n_b': len(b),
                'rate_a': a.mean(), 'rate_b': b.mean(),
                'difference_pp': 100 * (a.mean() - b.mean()), 'p_value': p_value,
            })

        table = pd.DataFrame(results).set_index('hypothesis')
        p = table['p_value'].dropna().sort_values()
        # Холм с поправкой на две запланированные гипотезы.
        table['p_holm'] = (p * (len(comparisons) - np.arange(len(p)))).cummax().clip(upper=1)
        table['reject_h0'] = table['p_holm'].lt(.05)
        return table

    def _phik_to_target(self, users, target, include_gap=False):
        features = self.FIRST_FEATURES + [
            'first_month', 'avg_revenue_rub', 'avg_revenue_per_ticket_rub',
            'avg_tickets', 'observed_span_days', 'observation_days',
        ]
        if include_gap:
            features.append('avg_days_between')
        data = users[features + [target]]
        varying = data.columns[data.nunique().gt(1)]
        if target not in varying:
            return pd.Series(np.nan, index=features, dtype=float)

        numeric = [
            'first_revenue_rub', 'first_tickets', 'avg_revenue_rub',
            'avg_revenue_per_ticket_rub', 'avg_tickets', 'observed_span_days',
            'observation_days', 'avg_days_between', 'total_orders',
        ]
        data = data[varying].copy()
        for col in data.select_dtypes('category'):
            data[col] = data[col].cat.remove_unused_categories()
        matrix = data.phik_matrix(
            interval_cols=[col for col in numeric if col in data],
            bins=10, quantile=True, njobs=1,
        )
        return matrix[target].reindex(features)

    def calculate_correlations(self, users):
        return pd.DataFrame({
            'Число заказов': self._phik_to_target(users, 'total_orders'),
            'Группы числа заказов': self._phik_to_target(users, 'order_segment'),
            'Только повторные': self._phik_to_target(
                users.loc[users['is_two']], 'total_orders', include_gap=True,
            ),
        })

