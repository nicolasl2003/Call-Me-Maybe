import numpy as np
import json


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

# def generate_string(model, prompt, definitions, name_function):


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
