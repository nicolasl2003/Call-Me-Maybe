def generate_params(parameters: dict, values: list) -> dict:
    param_names = list(parameters.keys())

    if (len(values) != len(param_names)):
        raise Exception(
            f"Expected {len(param_names)} parameters {param_names}, "
            f"got {len(values)} values {values}"
        )
    result: dict = {}
    for name, raw_value in zip(param_names, values):
        excepted_type = parameters[name]["type"]
        if (excepted_type == "number"):
            result[name] = float(raw_value)
        elif (excepted_type == "string"):
            result[name] = str(raw_value)
        elif (excepted_type == "boolean"):
            result[name] = str(raw_value).lower() in ("true", "1", "yes")
        else:
            result[name] = raw_value

    return (result)


def generate_json(prompt: str, name_function: str,
                  parameters: dict, values: list) -> dict:
    result: dict = {
        "prompt": f"{prompt}",
        "name": f"{name_function}",
        "parameters": generate_params(parameters, values)
    }
    return (result)
