import json
from llm_sdk import Small_LLM_Model
import numpy as np
from pathlib import Path
from generate_number import generate_number


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
        print("ok")
        if (param_type == "number"):
            print("ok4")
            value = generate_number(model, token, definitions, name)
            print("ok2")
            result = value.split(',')
            return (result)
        elif (param_type == "string"):
            pass
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


if (__name__ == "__main__"):
    model = Small_LLM_Model()
    prompt = parsing_prompt()
    function_text = ""
    params = []
    definitions = parsing_definition()
    for fn in definitions:
        function_text += f"- name : {fn['name']} description : {fn['description']}\n"
        valid_names = [d["name"] for d in definitions]
        print(f"\nNoms valides : {valid_names}")
    for token in prompt:
        prompt_function = (
            "<|im_start|>system\n"
            "You are a function-calling assistant. "
            "Choose the SINGLE most appropriate function for the user's request.\n"
            "Available functions:\n"
            f"{function_text}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{token}<|im_end|>\n"
            "<|im_start|>assistant\n"
            '{"name": "'
        )
        tokens = model.encode(prompt_function).tolist()
        name = generate_name(tokens, valid_names, model)
        selected_def = next(d for d in definitions if (d["name"] == name, None))
        parameter_schema = selected_def["parameters"]
        result = select_function(model, tokens, definitions, name, parameter_schema)
        print(token, end="\n\n")
        print(f"parametre : {result}")
        print(f"name : {name}")
