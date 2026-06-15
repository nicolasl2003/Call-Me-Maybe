import json
import sys


def parsing_definition() -> dict:
    with open("input/functions_definition.js", "r") as file:
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
    try:
        prompt: list[str] = parsing_prompt()
    except json.decoder.JSONDecodeError:
        print("Error !")
        sys.exit()
    print("prompte :")
    for i in prompt:
        print(i)
    print("definiton :")
