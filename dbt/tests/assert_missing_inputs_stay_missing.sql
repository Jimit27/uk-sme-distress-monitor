-- Regression guard: a ratio whose inputs are missing must be NULL, never a clip bound.
select company_number
from {{ ref('int_accounts__features') }}
where (cash is null and cash_to_assets is not null)
   or (current_liabilities is null and current_ratio is not null)
