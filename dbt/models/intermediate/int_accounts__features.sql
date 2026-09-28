-- Point-in-time features computed only from the filed accounts themselves.
-- Nothing here reads the register's current state, so every feature was
-- knowable on the day the accounts were filed.
with a as (
    select * from {{ ref('stg_ch__accounts') }}
),

base as (
    select
        a.*,
        coalesce(equity_cur, net_assets_cur)                                   as equity,
        coalesce(equity_prior, net_assets_prior)                               as equity_p,
        coalesce(creditors_within_1y_cur, current_assets_cur - net_current_assets_cur)       as current_liabilities,
        coalesce(creditors_within_1y_prior, current_assets_prior - net_current_assets_prior) as current_liabilities_p,
        coalesce(fixed_assets_cur, coalesce(ppe_cur, 0) + coalesce(intangibles_cur, 0), 0)       as fixed_total,
        coalesce(fixed_assets_prior, coalesce(ppe_prior, 0) + coalesce(intangibles_prior, 0), 0) as fixed_total_p
    from a
),

totals as (
    select
        *,
        case
            when current_assets_cur is not null then fixed_total + current_assets_cur
            when total_assets_less_cl_cur is not null and current_liabilities is not null
                then total_assets_less_cl_cur + current_liabilities
        end as total_assets,
        case
            when current_assets_prior is not null then fixed_total_p + current_assets_prior
            when total_assets_less_cl_prior is not null and current_liabilities_p is not null
                then total_assets_less_cl_prior + current_liabilities_p
        end as total_assets_p
    from base
)

select
    t.company_number,
    t.filing_batch,
    t.batch_type,
    t.filing_period_end,
    t.filing_ref_date,
    t.balance_sheet_date,
    t.entity_name,
    t.source_file,

    -- size
    total_assets,
    equity,
    ln(1 + greatest(coalesce(total_assets, 0), 0))                               as log_total_assets,
    ln(1 + greatest(coalesce(employees_cur, 0), 0))                              as log_employees,
    employees_cur                                                                as employees,
    n_officers,
    ln(1 + n_facts)                                                              as log_n_facts,

    -- solvency & leverage
    {{ safe_ratio('equity', 'total_assets') }}                                   as equity_to_assets,
    (equity < 0)::int                                                            as negative_equity,
    {{ safe_ratio('total_assets - equity', 'total_assets', 0, 20) }}             as liabilities_to_assets,
    {{ safe_ratio('creditors_after_1y_cur', 'total_assets', 0, 20) }}            as long_term_creditors_to_assets,

    -- liquidity
    {{ safe_ratio('current_assets_cur', 'current_liabilities', 0, 50) }}         as current_ratio,
    {{ safe_ratio('net_current_assets_cur', 'total_assets') }}                   as working_capital_to_assets,
    {{ safe_ratio('cash_cur', 'total_assets', 0, 1) }}                           as cash_to_assets,
    {{ safe_ratio('cash_cur', 'current_liabilities', 0, 50) }}                   as cash_to_current_liabilities,
    {{ safe_ratio('debtors_cur', 'total_assets', 0, 1) }}                        as debtors_to_assets,
    {{ safe_ratio('current_liabilities', 'total_assets', 0, 20) }}               as current_liabilities_to_assets,

    -- year-on-year movement
    {{ safe_ratio('equity - equity_p', 'greatest(total_assets, total_assets_p)') }}      as equity_change_to_assets,
    {{ safe_ratio('total_assets - total_assets_p', 'total_assets_p', -1, 10) }}          as asset_growth,
    {{ safe_ratio('cash_cur - cash_prior', 'greatest(total_assets, total_assets_p)') }}  as cash_change_to_assets,
    {{ safe_ratio('current_liabilities - current_liabilities_p', 'total_assets_p', -10, 10) }} as current_liabilities_change_to_assets,
    (equity < 0 and equity_p >= 0)::int                                          as equity_turned_negative,
    (equity_p is not null or total_assets_p is not null)::int                    as has_prior_year,

    -- filing behaviour
    is_dormant::int                                                              as is_dormant,
    (turnover_cur is not null)::int                                              as discloses_turnover,
    (filing_ref_date - balance_sheet_date)                                       as filing_lag_days,
    (filing_ref_date > balance_sheet_date + interval 9 month)::int               as filed_late,

    -- identity
    ag.number_prefix,
    case
        when ag.number_prefix = 'EW' then 'england_wales'
        when ag.number_prefix = 'SC' then 'scotland'
        when ag.number_prefix = 'NI' then 'northern_ireland'
        when ag.number_prefix in ('OC', 'SO', 'NC') then 'llp'
        else 'other'
    end                                                                          as entity_group,
    round(datediff('day', ag.approx_incorporation_date, t.balance_sheet_date) / 365.25, 2) as approx_age_years,

    -- raw figures kept for the app's company view
    current_assets_cur   as current_assets,
    current_liabilities,
    cash_cur             as cash,
    net_current_assets_cur as net_current_assets,
    equity_p             as equity_prior,
    total_assets_p       as total_assets_prior,
    turnover_cur         as turnover
from totals t
left join {{ ref('int_company_number_age') }} ag using (company_number)
