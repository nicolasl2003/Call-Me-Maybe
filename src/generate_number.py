import numpy as np
import json
import re


def filter_logits(logits, allowed_ids):
    filtered = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        filtered[token_id] = logits[token_id]
    return (filtered)


def get_param_info(definitions: list[dict], name_function: str,
                   current_param: str) -> dict:
    func_def = next((f for f in definitions if
                     (f["name"] == name_function)), {})
    return (func_def.get("parameters", {}).get(current_param, {}))


def generate_string(
    model,
    prompt: str,
    definitions: list[dict],
    name_function: str,
    current_param: str,
    filled_param: dict
) -> str:
    """
        Generate a value for a specific function-call parameter using greedy
        token-by-token decoding with the given model.

        Builds a type-aware prompt (number, boolean, regex, or generic string)
        and decodes tokens until a stopping condition is met (newline or
        repeated trailing characters), then returns the cleaned value.

        Args:
            model: Language model exposing `encode`, `decode`, and
                `get_logits_from_input_ids`.
            prompt (str): Natural language request
            describing the desired action.
            definitions (list[dict]): Available function definitions, used to
                look up parameter metadata (e.g. type).
            name_function (str): Name of the function being called.
            current_param (str): Name of the parameter to generate a value for.
            filled_param (dict): Parameters already filled, used as context.

        Returns:
            str: The generated value, stripped of quotes and whitespace.
    """
    param_info = get_param_info(definitions, name_function, current_param)
    param_type = param_info.get("type", "string")
    prompt_function = (
        "Fill the JSON function call for the request. "
        "Output ONLY the raw, exact, short value for the requested parameter. "
        "No quotes, no explanation, no extra text.\n"
        "Rules by type:\n"
        "  - number: a raw integer or float, no units.\n"
        "  - boolean: exactly 'true' or 'false'.\n"
        "  - string (regex) — ONLY applies when the parameter name is exactly "
        "'regex': output a PATTERN, NEVER copy literal values/numbers from "
        "the request:\n"
        "    Request: Replace all numbers in \"I have 5 cats\" with X -> regex: \\d+\n"
        "    Request: Replace all digits in \"a1b2\" with X -> regex: \\d+\n"
        "    Request: Replace all vowels in 'hello world' with * -> "
        "regex: [aeiouAEIOU]\n"
        "    Request: Replace all letters in 'abc123' with # -> regex: [a-zA-Z]+\n"
        "    Request: Replace the word 'cat' with 'dog' in 'the cat sat' -> regex: cat\n"
        "    Rule: to match ANY ONE of several characters (all vowels, all "
        "letters, ...) you MUST wrap them in square brackets, e.g. "
        "[aeiouAEIOU]. A bare sequence of letters with no brackets (aeiouAEIOU) "
        "is NOT a valid pattern and only matches that exact literal string.\n"
        "  - string (other) — used for every parameter NOT named 'regex' "
        "(e.g. replacement, source_string, name, s): a raw short value (word, "
        "name, sentence) copied exactly from the request, no quotes. NEVER "
        "output a regex pattern here.\n"
        "    Request: Replace all vowels in 'test' with asterisks, "
        "parameter 'replacement' -> replacement: *\n"
        "    Request: Replace every occurrence of 'x' with '#' in "
        "'exit code', parameter 'replacement' -> replacement: #\n"
        "    Request: Replace all numbers in \"I have 5 cats\" with NUMBERS, "
        "parameter 'replacement' -> replacement: NUMBERS\n"
        "    Rule: if the replacement is already given as a quoted literal "
        "(e.g. with '@'), copy that literal exactly. If instead it is "
        "described in words that name a symbol (asterisk/star -> *, "
        "hash/pound -> #, at sign -> @, dash/hyphen -> -), output that single "
        "character. Otherwise copy the exact word/phrase requested — it is "
        "almost never a number found elsewhere in the sentence.\n\n"
        f"Request: {prompt}\n"
        f"Function: {name_function}\n"
        f"Parameter '{current_param}' has type: {param_type}\n"
        f"Already filled parameters: {filled_param}\n"
        "Reminder: if this value names a symbol (asterisk/star -> *, "
        "hash/pound -> #, at sign -> @, dash/hyphen -> -), output only that "
        "single character, never the word itself.\n"
        f"Value for parameter '{current_param}':"
    )
    generated: list[int] = []
    tokens = model.encode(prompt_function).tolist()[0]

    for _ in range(50):
        logits = model.get_logits_from_input_ids(tokens + generated)
        next_token = int(logits.index(max(logits)))

        generated.append(next_token)

        decoded_stopping = model.decode(generated)
        if ("\n" in decoded_stopping):
            break
        if (len(decoded_stopping) >= 8 and len(set(decoded_stopping[-8:])) <= 1):
            break

    value = model.decode(generated).strip()
    value = value.split("\n")[0]
    value = value.strip().strip('"').strip("'")
    return (value)


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
