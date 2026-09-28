-- Health of the whole register by SIC section at the snapshot date.
with r as (
    select
        r.*,
        coalesce(s.sic_section, case when r.sic_code_1 in ('99999', '74990') then 'Z' else '?' end) as sic_section,
        coalesce(s.sic_section_name,
                 case when r.sic_code_1 = '99999' then 'Dormant company'
                      when r.sic_code_1 = '74990' then 'Non-trading company'
                      else 'Not supplied' end)                                   as sic_section_name,
        (r.status_group = 'active'
         and r.accounts_next_due < date '{{ var("outcome_snapshot_date") }}')::int as accounts_overdue
    from {{ ref('stg_ch__register') }} r
    left join {{ ref('sic_sections') }} s
        on r.sic_division between s.division_from and s.division_to
)

select
    sic_section,
    sic_section_name,
    count(*)                                                        as companies,
    sum((status_group = 'insolvency')::int)                         as in_insolvency,
    sum((status_group = 'strike_off_proposed')::int)                as strike_off_proposed,
    sum(accounts_overdue)                                           as accounts_overdue,
    round(100.0 * sum((status_group = 'insolvency')::int) / count(*), 3)          as insolvency_rate_pct,
    round(100.0 * sum((status_group = 'strike_off_proposed')::int) / count(*), 3) as strike_off_rate_pct,
    round(100.0 * sum(accounts_overdue) / count(*), 3)                            as overdue_rate_pct,
    round(median(datediff('day', incorporation_date, date '{{ var("outcome_snapshot_date") }}') / 365.25), 1) as median_age_years
from r
group by all
order by companies desc
