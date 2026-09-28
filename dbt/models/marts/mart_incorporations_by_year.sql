-- Survivorship view of the register: companies still on it by incorporation year
-- and what share of each vintage is now in a distress state.
select
    year(incorporation_date)                                                    as incorporation_year,
    count(*)                                                                    as companies,
    round(100.0 * sum((status_group = 'insolvency')::int) / count(*), 3)        as insolvency_rate_pct,
    round(100.0 * sum((status_group = 'strike_off_proposed')::int) / count(*), 3) as strike_off_rate_pct
from {{ ref('stg_ch__register') }}
where incorporation_date >= date '1990-01-01'
group by 1
order by 1
