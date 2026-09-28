-- The newest daily-filed accounts, ready to be scored by the trained model.
-- Enriched with register context (sector, location, current status) for the
-- dashboard only - the model never sees these columns.
with f as (
    select *
    from {{ ref('int_accounts__features') }}
    where batch_type = 'daily'
      and is_dormant = 0
      and total_assets > 0
    qualify row_number() over (partition by company_number order by filing_ref_date desc, balance_sheet_date desc) = 1
)

select
    f.*,
    r.company_status   as current_status,
    r.status_group     as current_status_group,
    r.sic_code_1,
    s.sic_section,
    s.sic_section_name,
    r.postcode_area,
    r.post_town,
    r.incorporation_date
from f
left join {{ ref('stg_ch__register') }} r using (company_number)
left join {{ ref('sic_sections') }} s
    on r.sic_division between s.division_from and s.division_to
