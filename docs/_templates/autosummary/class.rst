{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:

    {% block methods %}
    .. automethod:: __init__

    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
    {% for item in methods if item not in inherited_members %}
        ~{{ name }}.{{ item }}
    {%- endfor %}

    {% if inherited_members %}
    .. rubric:: {{ _('Inherited Methods') }}
    .. autosummary::
    {% for item in methods if item in inherited_members %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
