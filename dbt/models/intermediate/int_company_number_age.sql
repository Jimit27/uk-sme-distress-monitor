-- Approximate incorporation date from the company number alone.
--
-- Companies House allocates numbers sequentially within each prefix, so a
-- company's number places it in time. We look up the nearest *lower* number
-- still on the register and borrow its incorporation date. Using a neighbour
-- (never the company itself) means companies that have since left the
-- register get exactly the same treatment as survivors - otherwise
-- "missing incorporation date" would leak the outcome.
with accounts as (
    select distinct
        company_number,
        coalesce(nullif(regexp_extract(company_number, '^([A-Z]+)', 1), ''), 'EW') as number_prefix,
        try_cast(regexp_extract(company_number, '(\d+)$', 1) as bigint)          as number_seq
    from {{ ref('stg_ch__accounts') }}
),

register as (
    select
        coalesce(nullif(regexp_extract(company_number, '^([A-Z]+)', 1), ''), 'EW') as number_prefix,
        try_cast(regexp_extract(company_number, '(\d+)$', 1) as bigint)          as number_seq,
        incorporation_date
    from {{ ref('stg_ch__register') }}
    where incorporation_date is not null
)

select
    a.company_number,
    a.number_prefix,
    r.incorporation_date as approx_incorporation_date
from accounts a
asof left join register r
    on a.number_prefix = r.number_prefix
   and a.number_seq > r.number_seq
