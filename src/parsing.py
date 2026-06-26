import json
from llm_sdk import Small_LLM_Model
import numpy as np
import sys


def generate_name(tokens, valide_names, model, vocab) -> str:
    encoded_names = {}
    for name in valide_names:
        encoded_names[name] = model.encode(name).tolist()
    generated = []
    candidates = list(encoded_names.values())

    while True:
        pos = len(generated)
        valid_ids = set()
        for name_tokens in candidates:
            if (pos < len(name_tokens)):
                valid_ids.add(name_tokens[pos])
        logits = model.get_logits_from_input_ids(tokens[0])
        masked = np.full(len(logits), -np.inf)
        for id in valid_ids:
            masked[id] = logits[id]

        next_token = int(np.argmax(masked))
        generated.append(next_token)
        candidates = [
            name_tokens for name_tokens in candidates
            if (pos < len(name_tokens) and next_token)
        ]
        for name, name_tokens in encoded_names.items():
            if (name_tokens == generated):
                return (name)


def generate_constrained(prompt, definition, model, vocab) -> dict:
    tokens = model.encode(prompt).tolist()

    valid_names = [fn["name"] for fn in definition]
    name = generate_name(tokens, valid_names, model, vocab)

    fn = next(f for f in definition if (f["name"] == name))
    params = generate_params(tokens, fn["parameters"], model, vocab)

    return ({"name": name, "parameters": params})


def speak_model(prompt: list[str], definition: dict):
    prompt = f"<|im_start|>You are an assistant; your goal is to find the following function. {definition} according to the user prompt {prompt}, You need to give me the name of the function.n<|im_end|>\n{{\n\"name\":"
    model = Small_LLM_Model()
    output: str = ""
    tokens = model.encode(prompt).tolist()
    end = model.encode("<|endoftext|>")
    end = int(end[0])

    for a in range(1000):
        if (tokens[0][-1] == end):
            break
        logits = model.get_logits_from_input_ids(tokens[0])
        prob_logits = int(np.argmax(logits))
        tokens[0].append(prob_logits)
        output += model.decode(tokens[0])
        print(output)


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
    try:
        prompt: list[str] = parsing_prompt()
        definition: dict = parsing_definition()
    except (json.decoder.JSONDecodeError, FileNotFoundError):
        print("Error !")
        sys.exit()
    speak_model(prompt, definition)
