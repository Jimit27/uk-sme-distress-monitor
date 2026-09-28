-- Typed, cleaned register snapshot: one row per live company.
with src as (
    select * from {{ source('companies_house', 'register_snapshot') }}
),

typed as (
    select
        upper(trim(company_number))                                  as company_number,
        company_category,
        company_status,
        {{ status_group('company_status') }}                         as status_group,
        try_strptime(incorporation_date, '%d/%m/%Y')::date           as incorporation_date,
        try_strptime(dissolution_date, '%d/%m/%Y')::date             as dissolution_date,
        try_strptime(accounts_next_due, '%d/%m/%Y')::date            as accounts_next_due,
        try_strptime(accounts_last_made_up, '%d/%m/%Y')::date        as accounts_last_made_up,
        upper(accounts_category)                                     as accounts_category,
        try_strptime(confstmt_next_due, '%d/%m/%Y')::date            as confstmt_next_due,
        try_cast(mortgage_charges as integer)                        as mortgage_charges,
        try_cast(mortgage_outstanding as integer)                    as mortgage_outstanding,
        sic_code_1,
        try_cast(left(sic_code_1, 2) as integer)                     as sic_division,
        postcode_area,
        upper(post_town)                                             as post_town
    from src
    where company_number is not null
)

select *
from typed
qualify row_number() over (partition by company_number order by incorporation_date desc nulls last) = 1
