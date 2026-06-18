import json
from llm_sdk import Small_LLM_Model
import numpy as np
import sys


def speak_model(prompt: list[str], definition: dict):
    prompt = f"<|im_start|>You are an assistant; your goal is to find the following function. {definition} according to the user prompt {prompt}, You need to give me the name of the function.n<|im_end|>\n{{\n\"name\":"
    model = Small_LLM_Model()
    reponse: str = ""
    tokens = model.encode(prompt).tolist()
    end = model.encode("<|endoftext|>")
    end = int(end[0])

    for a in range(1000):
        if (tokens[0][-1] == end):
            break
        logits = model.get_logits_from_input_ids(tokens[0])
        prob_logits = int(np.argmax(logits))
        tokens[0].append(prob_logits)
        reponse += model.decode(tokens[0])
        print(reponse)


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
