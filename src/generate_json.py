def generate_json(prompt, name_function, definitions, values):
    item = next((d for d in definitions if (d["name"] == name_function)), None)
    if (item is None):
        raise Exception(f"No function definition found for '{name_function}'")
    param_defs = item["parameters"]
    param_name = list(param_defs.keys())

    if (len(values) != list(param_defs.keys())):
        raise Exception(
            f"Expected {len(param_name)} parameters for '{name_function}', "
            f"got {len(values)}: {values}"
        )
    parameters = {}
    for name, raw_values in zip(param_name, values):
        expected_type = param_defs[name]["type"]
        if (expected_type == "number"):
            parameters[name] == float(raw_values)
        elif (expected_type == "string"):
            parameters[name] = str(raw_values)
        elif (expected_type == "boolean"):
            parameters[name] = str(raw_values).lower() in ("true", "1", "yes")
        else:
            parameters[name] = raw_values

    result = {
        "prompt": prompt,
        "name": name_function,
        "parameters": parameters
    }
    return (result)
