{%- extends 'lab/index.html.j2' -%}

{%- block body %}
    {{ super() }}
    <script src="scrythe.js"></script>
{%- endblock body %}

{%- block header %}
    {{ super() }}
    <link rel="stylesheet" href="style.css">
{%- endblock header %}