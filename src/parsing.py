import json
from llm_sdk import Small_LLM_Model
import numpy as np
import sys


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

def parsing_definition() -> dict:
    with open("input/functions_definition.json", "r") as file:
        data = json.load(file)
    return (data)


def parsing_prompt() -> list[str]:

    prompt: list[str] = []
    with open("input/function_calling_tests.json", "r") as file:
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
    definitions = parsing_definition()

    for fn in definitions:
        function_text += f"- name : {fn['name']} description : {fn['description']}\n"
    test_prompt = prompt[0]
    valid_names = [d["name"] for d in definitions]
    print(f"Noms valides : {valid_names}\n")

    prompt = (
        "<|im_start|>system\n"
        "You are a function-calling assistant. "
        "Choose the SINGLE most appropriate function for the user's request.\n"
        "Available functions:\n"
        f"{function_text}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        '{"name": "'
    )

    tokens = model.encode(prompt).tolist()

    name = generate_name(tokens, valid_names, model)
    print(f"Nom Genere : {name}")
