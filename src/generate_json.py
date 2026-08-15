def generate_params(parameters: dict, values: list) -> dict:
    """Build a dictionary of typed parameter values.

    Zips `parameters` (mapping parameter names to their schema info,
    including an expected "type") with `values` (the raw extracted
    values, in the same order), converting each raw value to the
    appropriate Python type ("number" -> float, "string" -> str,
    "boolean" -> bool), based on the corresponding schema's "type".

    Args:
        parameters: A mapping of parameter names to their schema
            info, each containing at least a "type" key.
        values: A list of raw values (e.g. strings) to convert and
            assign to the parameters, in the same order as
            `parameters.keys()`.

    Returns:
        A dictionary mapping each parameter name to its value,
        converted to the type specified in its schema.

    Raises:
        Exception: If the number of `values` does not match the
            number of parameters.
    """
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
            result[name] = bool(raw_value).lower() in ("true", "1", "yes")
        else:
            result[name] = raw_value
    return (result)


def generate_json(prompt: str, name_function: str,
                  parameters: dict, values: list) -> dict:
    """Assemble the final structured result for a function call.

    Wraps the original prompt, the detected function name, and the
    typed parameter values (built via `generate_params`) into a
    single dictionary describing the resolved function call.

    Args:
        prompt: The original user request.
        name_function: The name of the function that was matched.
        parameters: A mapping of parameter names to their schema
            info (used to type-convert the raw `values`).
        values: A list of raw values, in the same order as
            `parameters.keys()`, to be converted and included in the
            result.

    Returns:
        A dictionary with the keys "prompt", "name", and
        "parameters", the latter containing the typed parameter
        values.
    """
    result: dict = {
        "prompt": f"{prompt}",
        "name": f"{name_function}",
        "parameters": generate_params(parameters, values)
    }
    return (result)
