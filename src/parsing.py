import json
from pathlib import Path
from .generate_number import (generate_number,
                              generate_string
                              )


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


def generate_name(tokens, valid_names, model) -> str:
    encoded_names = {}
    generated = []

    for name in valid_names:
        encoded_names[name] = model.encode(name).tolist()[0]
    while (True):
        candidates = [
            enc for enc in encoded_names.values()
            if (enc[:len(generated)] == generated)
        ]
        allowed = set()
        for enc in candidates:
            if (len(enc) > len(generated)):
                allowed.add(enc[len(generated)])
        if (not allowed):
            break
        logits = model.get_logits_from_input_ids(tokens[0])
        best = max(allowed, key=lambda tok_id: logits[tok_id])
        generated.append(best)

    return (model.decode(generated))


def select_function(model, token, definitions, name, parameter_schema):
    result = []

    for param_name, paran_info in parameter_schema.items():
        param_type = paran_info["type"]
        if (param_type == "number"):
            value = generate_number(model, token, definitions, name)
            result = value.split(',')
            return (result)
        elif (param_type == "string"):
            value = generate_string(model, token, definitions, name)
            result = value
            return (result)
        else:
            print(f"Type not supporte : {param_type} for {param_name}")
            value = None
    return (result)


def parsing_definition() -> dict:
    with open(INPUT_DIR / "functions_definition.json", "r") as file:
        data = json.load(file)
    return (data)


def parsing_prompt() -> list[str]:

    prompt: list[str] = []
    with open(INPUT_DIR / "function_calling_tests.json", "r") as file:
        data: dict = json.load(file)

    for i in data:
        try:
            if (i['prompt'] == ""):
                print("Error : prompte is empty")
            prompt.append(i['prompt'])
        except KeyError:
            print("clee is empty")
            return
    return (prompt)
