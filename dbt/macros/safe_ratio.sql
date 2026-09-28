{#- Ratio that returns NULL on a zero/NULL denominator and clips outliers.
    Filed accounts contain plenty of tiny denominators (a company with GBP 1
    of assets), so unclipped ratios would be dominated by noise. -#}
{#- NB: DuckDB's least/greatest skip NULLs, so the NULL case is handled
    explicitly - otherwise a missing ratio would silently become the bound. -#}
{% macro safe_ratio(numerator, denominator, lo=-10, hi=10) -%}
    case
        when ({{ numerator }}) is null or ({{ denominator }}) is null or ({{ denominator }}) = 0 then null
        else least(greatest(({{ numerator }}) / ({{ denominator }}), {{ lo }}), {{ hi }})
    end
{%- endmacro %}

{% macro status_group(status_col) -%}
    case
        when {{ status_col }} is null then 'removed'
        when regexp_matches(lower({{ status_col }}), 'liquidation|administration|receiver|voluntary arrangement|insolvency') then 'insolvency'
        when lower({{ status_col }}) like '%proposal to strike off%' then 'strike_off_proposed'
        when lower({{ status_col }}) = 'active' then 'active'
        else 'other'
    end
{%- endmacro %}
