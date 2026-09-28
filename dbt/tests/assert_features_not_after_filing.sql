-- Point-in-time guard: no training row may use a balance sheet dated after it was filed.
select company_number, balance_sheet_date, filing_period_end
from {{ ref('fct_training_cohort') }}
where balance_sheet_date > filing_period_end
