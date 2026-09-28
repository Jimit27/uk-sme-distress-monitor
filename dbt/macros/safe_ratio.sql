{#- Ratio that returns NULL on a zero/NULL denominator and clips outliers.
    Filed accounts contain plenty of tiny denominators (a company with GBP 1
    of assets), so unclipped ratios would be dominated by noise. -#}
{% macro safe_ratio(numerator, denominator, lo=-10, hi=10) -%}
    least(greatest(({{ numerator }}) / nullif(({{ denominator }}), 0), {{ lo }}), {{ hi }})
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
