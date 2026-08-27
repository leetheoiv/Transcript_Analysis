def get_template_params(row, fields:list=[]):
    """
    Maps selected row values to Jinja template parameters dynamically.
    """
    return {field: row.get(field) for field in fields}