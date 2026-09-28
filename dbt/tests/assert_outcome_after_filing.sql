-- Labels must be observed strictly after the features were available.
select company_number, horizon_days
from {{ ref('fct_training_cohort') }}
where horizon_days <= 0
