import json
from llm_sdk import Small_LLM_Model
import numpy as np


def filter_logits(logits, allowed_ids):
    filtered = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        filtered[token_id] = logits[token_id]
    return (filtered)


def get_digit_token_ids(model):
    with open(model.get_path_to_vocab_file()) as file:
        vocab = json.load(file)
    digits_ids = []

    for number in "0123456789.":
        digits_ids.append(vocab[number])
    return (digits_ids)


def generate_number(model, prompt, definitions, name_function):
    generated = []
    count_while = 0
    allowed_digit = get_digit_token_ids(model)
    logits_comma = int(model.encode(',')[0][0])
    logits_point = int(model.encode('.')[0][0])
    logits_brace = int(model.encode('}')[0][0])
    item = [i for i in definitions if i["name"] == name_function][0]
    if item is None:
        raise Exception("No item found with function")
    nb_params = len(item['parameters'])
    MAX_ITER = 50
    prompt_function = (
        "<|im_start|>system\n"
        "You are a strict text-extraction assistant. "
        "Your ONLY job is to copy the numbers that literally appear in the user's sentence, "
        "in the order they appear, as digits separated by commas.\n"
        "You are NOT a calculator. NEVER compute, solve, or transform any number. "
        "NEVER output a number that is not literally written in the user's text. "
        "Always end your output immediately after the last digit, followed by a closing brace '}'.\n"
        "\n"
        "Examples:\n"
        "User: What is the sum of 5 and 7?"
        "Output: 5,7}\n"
        "\n"
        "User: What is the square root of 16?"
        "Output: 16}\n"
        "\n"
        "User: Calculate the square root of 144"
        "Output: 144}\n"
        "\n"
        "User: What is the sum of 2.5 and 3.5"
        "Output: 2.5,3.5}\n"
        "User: What is the sum of 230.5 and 404.5"
        "Output: 230.5,404.5}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Output: "
    )

    tokens = model.encode(prompt_function).tolist()
    allowed = allowed_digit + [logits_comma, logits_brace, logits_point]
    while (True):
        count_while += 1
        chosen_card = model.get_logits_from_input_ids(tokens[0] + generated)
        nb_comma = generated.count(logits_comma)
        if (generated and generated[-1] == logits_comma):
            allowed_now = allowed_digit
        elif (nb_comma + 1 >= nb_params):
            allowed_now = allowed_digit + [logits_brace]
        else:
            allowed_now = allowed
        result_card = filter_logits(chosen_card, allowed_now)
        best = int(np.argmax(result_card))
        if (best == logits_brace):
            break
        if (count_while >= MAX_ITER):
            break
        generated.append(best)
    return (model.decode(generated))


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


def select_function(model, token, definitions, name):
    result = []

    for param_name, paran_info in parameter_schema.items():
        param_type = paran_info["type"]
        print("ok")
        if (param_type == "number"):
            value = generate_number(model, token, definitions, name)
            result = value.split(',')
        elif (param_type == "string"):
            pass
        else:
            print(f"Type not supporte : {param_type} for {param_name}")
            value = None
    return (result)


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
        params = select_function(model, tokens, definitions, name)
        print(token, end="\n\n")
        print(f"parametre : {params}")
        print(f"name : {name}")
