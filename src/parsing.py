import json
from llm_sdk import Small_LLM_Model
import numpy as np
import sys


def filter_logits(logits, allowed_ids):
    filtered = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        filtered[token_id] = logits[token_id]
    return (filtered)


def get_digit_token_ids(model):
    with open(model.get_path_to_vocab_file()) as file:
        vocab = json.load(file)
    digits_ids = []

    for number in "0123456789":
        digits_ids.append(vocab[number])
    return (digits_ids)


def generate_number(model, prompt, definitions, name_function):
    generated = []
    count_while = 0
    allowed_digit = get_digit_token_ids(model)
    logits_comma = int(model.encode(',')[0][0])
    logits_brace = int(model.encode('}')[0][0])
    prompt_function = (
        "<|im_start|>system\n"
        "You are a function-calling assistant. "
        "Extract the multiple numbers from the user prompt.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        '"parameters": {"a":}'
    )
    tokens = model.encode(prompt_function).tolist()
    allowed = allowed_digit + [logits_comma]
    while (True):
        count_while += 1
        chosen_card = model.get_logits_from_input_ids(tokens[0] + generated)
        result_card = filter_logits(chosen_card, allowed)
        best = int(np.argmax(result_card))
        item = [i for i in definitions if i["name"] == name_function][0]
        if item is None:
            raise Exception("No item found with function")
        if (best == logits_brace or count_while >= len(item['parameters']) + 2):
            break
        generated.append(best)
    return (model.decode(generated))


'''def generate_param(model):
    with open(model.get_path_to_vocab_file()) as file:
        vocab = json.load(file)

    items = list(vocab.items())[:20]
    for k, v in items:
        print(repr(k), "->", v)
    print("\n Nombre total de token :", len(vocab))'''


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
    params = []
    definitions = parsing_definition()

    for fn in definitions:
        function_text += f"- name : {fn['name']} description : {fn['description']}\n"
    test_prompt = prompt[0]
    valid_names = [d["name"] for d in definitions]
    print(f"Noms valides : {valid_names}\n")

    prompt_function = (
        "<|im_start|>system\n"
        "You are a function-calling assistant. "
        "Choose the SINGLE most appropriate function for the user's request.\n"
        "Available functions:\n"
        f"{function_text}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{test_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        '{"name": "'
    )

    tokens = model.encode(prompt_function).tolist()
    name = generate_name(tokens, valid_names, model)
    params = generate_number(model, prompt[0], definitions, name)
    params_split = params.split(',')
    print(f"parametre : {params_split[0]}")
    print(f"name : {name}")
