-- Health of the register by postcode area (e.g. CR = Croydon, M = Manchester).
with r as (
    select
        *,
        (status_group = 'active' and accounts_next_due < date '{{ var("outcome_snapshot_date") }}')::int as accounts_overdue
    from {{ ref('stg_ch__register') }}
    where postcode_area is not null
),

town as (
    select postcode_area, mode(post_town) as main_town
    from r
    group by 1
)

select
    r.postcode_area,
    t.main_town,
    count(*)                                                                    as companies,
    sum((status_group = 'insolvency')::int)                                     as in_insolvency,
    round(100.0 * sum((status_group = 'insolvency')::int) / count(*), 3)        as insolvency_rate_pct,
    round(100.0 * sum((status_group = 'strike_off_proposed')::int) / count(*), 3) as strike_off_rate_pct,
    round(100.0 * sum(accounts_overdue) / count(*), 3)                          as overdue_rate_pct
from r
join town t using (postcode_area)
group by all
having count(*) >= 1000
order by companies desc
