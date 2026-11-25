from pathlib import Path

import pandas as pd

from src.processing import DataProcessor


def main():
    processor = DataProcessor(Path(__file__).resolve().parent / 'data')
    raw = processor.load_orders()
    rates = processor.load_rates()
    orders, cleaning = processor.preprocess(raw, rates)
    raw_dates = pd.to_datetime(raw['order_dt'])
    users, filtering = processor.create_user_profiles(orders, raw_dates.max().normalize())
    segments = processor.calculate_retention(users)
    hypotheses = processor.test_hypotheses(users, segments)
    revenue = processor.summarize_revenue(users)
    weekday_test = processor.analyze_weekdays(users)
    correlations = processor.calculate_correlations(users)

    total_orders = users['total_orders'].sum()
    mean_revenue = users['total_revenue_rub'].sum() / total_orders
    print('Отчёт по повторным покупкам на Яндекс Афише')
    print(f'Период {raw_dates.min():%d.%m.%Y} — {raw_dates.max():%d.%m.%Y}')
    print(f'Исходных заказов {len(raw):,}. Тенге переведены в рубли по курсу на дату заказа.')
    print(
        f'Удалено дублей {cleaning["duplicates"]:,} и заказов дороже p99 '
        f'({cleaning["revenue_p99"]:,.2f} ₽) {cleaning["expensive_orders"]:,}.'
    )
    print(
        f'Исключено профилей с отрицательной средней выручкой '
        f'{filtering["negative_profiles"]:,} и сверхактивных профилей '
        f'{filtering["active_profiles"]:,}.'
    )
    print(f'Пользователей {len(users):,}, заказов {total_orders:,}.')
    print(f'Доля с 2+ заказами {users["is_two"].mean():.2%}, средняя выручка {mean_revenue:,.2f} ₽ за заказ.')
    for label, count in users['order_segment'].value_counts(sort=False).items():
        print(f'  {label} — {count:,} пользователей ({count / len(users):.1%})')

    for title, field, limit in [
        ('Устройство', 'first_device', 2),
        ('Тип мероприятия', 'first_event_type', 7),
        ('Регион', 'first_region', 3),
    ]:
        print(f'\n{title} первого заказа')
        for segment, row in segments[field].head(limit).iterrows():
            print(f'  {segment} — {row.return_rate:.2%} с 2+ заказами (n={row.users:,.0f})')

    print('\nВыручка и число билетов по группам')
    for returned, row in revenue.iterrows():
        label = '2+ заказа' if returned else '1 заказ'
        print(
            f'  {label} — медиана выручки с заказа {row.median_order:,.2f} ₽ '
            f'(25–75-й перцентили {row.q25_order:,.2f}–{row.q75_order:,.2f} ₽), '
            f'медиана с билета {row.median_ticket:,.2f} ₽, '
            f'в среднем {row.avg_tickets:.2f} билета в заказе.'
        )

    weekdays = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    weekday = segments['first_weekday'].sort_index()
    print('\nДень недели первого заказа')
    for day, row in weekday.iterrows():
        print(f'  {weekdays[day]} — {row.return_rate:.2%} с 2+ заказами (n={row.users:,.0f})')
    rates_by_day = weekday['return_rate']
    print(f'Размах долей {100 * (rates_by_day.max() - rates_by_day.min()):.2f} п.п.')
    if weekday_test is not None:
        p_value, cramers_v = weekday_test
        print(f'Разведочная проверка χ², p = {p_value:.4g}, V Крамера = {cramers_v:.3f}.')

    print('\nМесяц первого заказа и доступное время наблюдения')
    for month, row in segments['first_month'].sort_index().iterrows():
        print(
            f'  {month} — {row.return_rate:.2%} с 2+ заказами (n={row.users:,.0f}), '
            f'медиана окна {row.median_observation_days:g} дн.'
        )
    print('Доля с 2+ заказами рассчитана за доступный период наблюдения.')

    print('\nПроверка гипотез')
    for name, row in hypotheses.iterrows():
        if pd.isna(row.p_holm):
            print(f'  {name} — для сравнения нужны обе группы.')
            continue
        conclusion = 'поддержана' if row.reject_h0 else 'подтверждения нет'
        print(
            f'  {name} — {row.rate_a:.2%} и {row.rate_b:.2%}, '
            f'p после Холма {row.p_holm:.4g}, гипотеза: {conclusion}.'
        )

    grouped = correlations['Группы числа заказов'].dropna()
    first = grouped.reindex(processor.FIRST_FEATURES).dropna()
    print('\nΦK с группой числа заказов')
    for title, scores in [
        ('Признаки первого заказа', first.nlargest(3)),
        ('Все признаки', grouped.nlargest(3)),
    ]:
        if not scores.empty:
            print(f'  {title} — ' + ', '.join(f'{col} {score:.3f}' for col, score in scores.items()))
    if grouped.empty:
        print('  В выборке недостаточно вариации для расчёта.')
    gap = correlations.loc['avg_days_between', 'Только повторные']
    if pd.notna(gap):
        print(f'  ΦK между средним интервалом и числом заказов у повторных покупателей {gap:.3f}.')
    print(
        '\nДля сравнения когорт используйте долю повторных покупок за 30 дней '
        'при полном окне наблюдения. Персональные рекомендации и изменения '
        'мобильного сценария проверяйте в A/B-тестах.'
    )


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f'Ошибка обработки данных — {exc}') from None