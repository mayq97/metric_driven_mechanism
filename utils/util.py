
def convert_config_to_dict(config_var):
    data = {}
    for var in dir(config_var):
        if not var.startswith("_"):
            data[var] = config_var.__getattribute__(var) if not config_var.__getattribute__(var) is None else "none"
    return data



