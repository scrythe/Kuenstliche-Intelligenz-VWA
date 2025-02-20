{%- extends 'markdown/index.md.j2' -%}

{% block any_cell %}
    {%- if 'image' not in cell.metadata.tags | default([]) -%}
        {{ super() }}
    {%- endif -%}
{% endblock %}