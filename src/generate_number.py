import numpy as np
import json
import re


def get_allowed_regex_tokens(model) -> list[int]:
    regex_chars: list = list("abcdefghijklmnopqrstuvwxy"
                             "zABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    regex_symbols: list[str] = ["[", "]", "^", "-", "\\", "s", "d", "w",
                                ".", "*", "+", "?", "(", ")", "|"]

    allowed_ids = set()
    for ch in regex_chars + regex_symbols:
        ids = model.encode(ch).tolist()[0]
        allowed_ids.update(ids)

    return list(allowed_ids)


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
    param_name: str,
    name_function: str,
) -> str:
    generated: list[int] = []
    i = 0
    MAX_ITER = 50

    prompt_function = model.encode(
        "You are a function-calling assistant. Extract the exact string value "
        "for the requested parameter from the user's request. "
        "If the parameter is a regex pattern, produce a valid regular expression "
        "that matches the described concept. "
        "Always end your output immediately after the last character, followed by a closing apostrophe '.\n"

        "Example 1:\n"
        "Request: \"Reverse the string 'hello world'\"\n"
        "Function: fn_reverse_string\n"
        "Parameter: s\n"
        "Answer: hello world\n\n"

        "Example 2:\n"
        "Request: \"Greet shrek\"\n"
        "Function: fn_greet\n"
        "Parameter: name\n"
        "Answer: shrek\n\n"

        "Example 3:\n"
        "Request: \"Replace all vowels in 'Programming is fun' with asterisks\"\n"
        "Function: fn_substitute_string_with_regex\n"
        "Parameter: source_string\n"
        "Answer: Programming is fun\n\n"

        "Example 4:\n"
        "Request: \"Replace all vowels in 'Programming is fun' with asterisks\"\n"
        "Function: fn_substitute_string_with_regex\n"
        "Parameter: regex\n"
        "Answer: [aeiouAEIOU]\n\n"

        "Example 5:\n"
        "Request: \"Replace all vowels in 'Programming is fun' with asterisks\"\n"
        "Function: fn_substitute_string_with_regex\n"
        "Parameter: replacement\n"
        "Answer: *\n\n"

        "Example 6:\n"
        "Request: \"Replace all digits in 'abc123' with hashtags\"\n"
        "Function: fn_substitute_string_with_regex\n"
        "Parameter: regex\n"
        "Answer: [0-9]\n\n"

        "Example 7:\n"
        "Request: \"Replace all digits in 'abc123' with hashtags\"\n"
        "Function: fn_substitute_string_with_regex\n"
        "Parameter: replacement\n"
        "Answer: #\n\n"

        f"Now answer for this request:\n"
        f"Request: \"{prompt}\"\n"
        f"Function: {name_function}\n"
        f"Parameter: {param_name}\n"
        f"Answer:"
    ).tolist()[0]

    finish = int(model.encode("<|endoftext|>").tolist()[0][0])
    logits_apostrophe = int(model.encode("'").tolist()[0][0])
    logits_returnline = int(model.encode('\n').tolist()[0][0])

    if (param_name == "regex"):
        allowed_ids = get_allowed_regex_tokens(model)
    else:
        allowed_ids = get_allowed_string_tokens(model, prompt)
    allowed_ids = list(set(allowed_ids + [logits_apostrophe]))

    while True:
        logits = model.get_logits_from_input_ids(prompt_function + generated)
        filtered = filter_logits(np.array(logits), allowed_ids)
        best = int(np.argmax(filtered))
        if ((best == finish) or (best == logits_apostrophe) or
                (best == logits_returnline)):
            break
        if i >= MAX_ITER:
            break
        generated.append(best)
        i += 1

    if (not generated):
        return ""

    return (model.decode(generated).strip())


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
    logits_negative = int(model.encode('-')[0][0])
    logits_negative_space: int = int(model.encode(' -')[0][0])
    item = [i for i in definitions if i["name"] == name_function][0]
    number_in_prompt = re.findall(r'-?\d+(?:\.\d+)?', prompt)
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

        is_start_of_number = ((not generated) or generated[-1] == logits_comma)

        if (is_start_of_number and nb_comma < len(number_in_prompt)
                and number_in_prompt[nb_comma].startswith('-')):
            generated.append(logits_negative)
            if (count_while >= MAX_ITER):
                break
            continue

        if (is_start_of_number):
            if (nb_comma + 1 >= nb_params):
                allowed_now = allowed_digit + [logits_negative,
                                               logits_negative_space,
                                               logits_brace]
            else:
                allowed_now = allowed_digit + [logits_negative,
                                               logits_negative_space]
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
