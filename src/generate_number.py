import numpy as np
import json


def get_allowed_string_tokens(model, prompt: str) -> list[int]:
    prompt_tokens = model.encode(prompt).tolist()[0]
    return list(set(prompt_tokens))


def filter_logits(logits, allowed_ids):
    filtered = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        filtered[token_id] = logits[token_id]
    return (filtered)


def generate_string(
    model,
    prompt: str,
    definitions: list[dict],
    name_function: str,
) -> str:
    generated: list[int] = []
    i = 0
    MAX_ITER = 50

    prompt_function = model.encode(
        "<|im_start|>system\n"
        "You are a strict text-extraction assistant. "
        "Your ONLY job is to copy the string parameter that literally appears "
        "in the user's sentence, exactly as written, character by character, "
        "including any apostrophes if they are part of the value.\n"
        "You are NOT allowed to translate, rephrase, correct, remove punctuation, or invent any part of it. "
        "NEVER output a value that is not literally written in the user's text. "
        "Always end your output immediately after the last character, followed by a closing apostrophe '.\n"
        "\n"
        "Examples:\n"
        "User: Reverse the string 'hello'\n"
        "Output: hello}\n"
        "\n"
        "User: Greet shrek\n"
        "Output: shrek}\n"
        "\n"
        "User: Say 'bonjour' to marie\n"
        "Output: bonjour}\n"
        "\n"
        "User: Greet john\n"
        "Output: john}\n"
        "\n"
        "User: Reverse the string 'it's fine'\n"
        "Output: it's fine}\n"
        "\n"
        "User: Say 'don't worry' to paul\n"
        "Output: don't worry}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Output: "
    ).tolist()[0]

    finish = int(model.encode("<|endoftext|>").tolist()[0][0])
    logits_apostrophe = int(model.encode('\'').tolist()[0][0])

    allowed_ids = get_allowed_string_tokens(model, prompt)
    allowed_ids = list(set(allowed_ids + [logits_apostrophe]))

    while True:
        logits = model.get_logits_from_input_ids(prompt_function + generated)
        filtered = filter_logits(np.array(logits), allowed_ids)
        best = int(np.argmax(filtered))

        if best == finish or best == logits_apostrophe:
            break
        if i >= MAX_ITER:
            break

        generated.append(best)
        i += 1

    if (not generated):
        return ""

    return (model.decode(generated))


def get_digit_token_ids(model):
    with open(model.get_path_to_vocab_file()) as file:
        vocab = json.load(file)
    digits_ids = []

    for number in "-0123456789.":
        digits_ids.append(vocab[number])
    return (digits_ids)


def generate_number(model, prompt, definitions, name_function):
    generated = []
    count_while = 0
    allowed_digit = get_digit_token_ids(model)
    logits_comma = int(model.encode(',')[0][0])
    logits_point = int(model.encode('.')[0][0])
    logits_brace = int(model.encode('}')[0][0])
    logits_negative = int(model.encode('-')[0][0])
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
        "User: What is the sum of 5 and 7?\n"
        "Output: 5,7}\n"
        "\n"
        "User: What is the square root of 16?\n"
        "Output: 16}\n"
        "\n"
        "User: Calculate the square root of 144"
        "Output: 144}\n"
        "\n"
        "User: What is the sum of 2.5 and 3.5\n"
        "Output: 2.5,3.5}\n"
        "User: What is the sum of 230.5 and 404.5\n"
        "Output: 230.5,404.5}\n"
        "User: What is the sum of -15 and 42?\n"
        "Output: -15,42}\n"
        "User: What is the sum of -3 and 8?\n"
        "Output: -3,8}\n"
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
            allowed_now = allowed_digit + [logits_negative]
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
