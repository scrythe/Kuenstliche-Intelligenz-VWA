{%- extends 'lab/index.html.j2' -%}


{% block any_cell %}
{% if 'hide_input' in cell['metadata'].get('tags', []): %}
    {%- block output_group -%}
        {{ super() }}
    {%- endblock output_group %}
{% else %}
    {{ super() }}
{% endif %}
{% endblock any_cell %}