-- Parsed accounts filings, typed and de-duplicated to one filing per company per batch.
-- Amended or duplicate filings in the same batch keep the latest balance sheet,
-- then the richest document (most tagged facts).
with src as (
    select * from {{ source('companies_house', 'accounts') }}
),

cleaned as (
    select
        upper(trim(company_number))                                  as company_number,
        balance_sheet_date,
        filing_batch,
        filing_period_end,
        -- Monthly batches only tell us the month of filing; use its midpoint.
        case when regexp_matches(filing_batch, '^\d{4}-\d{2}-\d{2}$')
             then 'daily' else 'monthly' end                         as batch_type,
        case when regexp_matches(filing_batch, '^\d{4}-\d{2}-\d{2}$')
             then filing_period_end
             else filing_period_end - interval 15 day end::date      as filing_ref_date,
        source_file,
        file_format,
        n_facts,
        n_officers,
        coalesce(is_dormant, false)                                  as is_dormant,
        entity_name,
        * exclude (company_number, balance_sheet_date, filing_batch, filing_period_end,
                   source_file, file_format, parse_ok, n_facts, n_officers, is_dormant, entity_name)
    from src
    where parse_ok
      and company_number is not null
      and balance_sheet_date is not null
)

select *
from cleaned
qualify row_number() over (
    partition by company_number, filing_batch
    order by balance_sheet_date desc, n_facts desc, source_file
) = 1
