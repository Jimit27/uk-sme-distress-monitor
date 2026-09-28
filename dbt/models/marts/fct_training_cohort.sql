-- Modelling table: accounts filed in the study months, labelled with what had
-- happened to each company by the outcome snapshot (~12 months later).
--
--   label_insolvency : company is in a formal insolvency process
--                      (liquidation, administration, receivership, CVA)
--   label_failure    : insolvency, OR a pending proposal to strike off, OR the
--                      company has already left the register
--
-- Population: trading companies with a usable balance sheet. Dormant
-- companies are excluded - they carry no credit risk and their voluntary
-- strike-offs would swamp the failure label.
{% set train = var('train_batches') %}
{% set test = var('test_batches') %}

with f as (
    select *
    from {{ ref('int_accounts__features') }}
    where filing_batch in ({% for b in train + test %}'{{ b }}'{% if not loop.last %}, {% endif %}{% endfor %})
),

-- a company filing in both months keeps only its earliest (training) filing,
-- so train and test never share a company
one_per_company as (
    select *
    from f
    qualify row_number() over (partition by company_number order by filing_ref_date, balance_sheet_date desc) = 1
),

outcome as (
    select company_number, company_status, status_group, sic_code_1, postcode_area
    from {{ ref('stg_ch__register') }}
)

select
    f.*,
    case when f.filing_batch in ({% for b in test %}'{{ b }}'{% if not loop.last %}, {% endif %}{% endfor %})
         then 'test' else 'train' end                                   as split,
    date '{{ var("outcome_snapshot_date") }}'                           as outcome_date,
    date '{{ var("outcome_snapshot_date") }}' - f.filing_ref_date        as horizon_days,
    coalesce(o.status_group, 'removed')                                 as outcome_group,
    o.company_status                                                    as outcome_status_text,
    (coalesce(o.status_group, 'removed') = 'insolvency')::int           as label_insolvency,
    (coalesce(o.status_group, 'removed') in ('insolvency', 'strike_off_proposed', 'removed'))::int as label_failure
from one_per_company f
left join outcome o using (company_number)
where f.is_dormant = 0
  and f.total_assets > 0
