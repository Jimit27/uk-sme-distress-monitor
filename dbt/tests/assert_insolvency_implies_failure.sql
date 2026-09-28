-- The narrow label must be a subset of the broad one.
select company_number
from {{ ref('fct_training_cohort') }}
where label_insolvency = 1 and label_failure = 0
