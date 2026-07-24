import json
from pathlib import Path
from .generate_number import (generate_number,
                              generate_string
                              )
from pydantic import BaseModel, model_validator, ValidationError
import sys

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


class Prompt(BaseModel):
    valid_prompt: str

    @model_validator(mode="after")
    def check_prompt(self) -> "Prompt":
        if (self.prompt == ""):
            raise ValueError(
                "Error: prompt is empty"
            )
        return (self)

    @classmethod
    def pars_list(cls, raw_data: list[dict]) -> list["Prompt"]:
        valid_prompts: list["Prompt"] = []
        for item in raw_data:
            try:
                p = cls.model_validator(item)
            except ValidationError as e:
                print(f"Error skipping item {item}: {e}")
                continue
            valid_prompts.append(p)
            return (valid_prompts)


class definition(BaseModel):
    name: str
    description: str
    parameters: dict



def select_function(model, token, definitions, name, parameter_schema):
    result = []

    for param_name, paran_info in parameter_schema.items():
        param_type = paran_info["type"]
        if (param_type == "number"):
            value = generate_number(model, token, definitions, name)
            result = value.split(',')
            return (result)
        elif (param_type == "string"):
            value = generate_string(model, token, definitions, name,
                                    parameter_schema)
            result = [value]
            return (result)
        else:
            print(f"Type not supporte : {param_type} for {param_name}")
            value = None
    return (result)


def parsing_definition() -> dict:
    try:
        with open(INPUT_DIR / "functions_definition.json", "r") as file:
            data = json.load(file)
    except json.decoder.JSONDecodeError:
        print("Error")
        sys.exit(1)
    return (data)


def parsing_prompt() -> list[str]:

    prompt: list[str] = []
    try:
        with open(INPUT_DIR / "function_calling_tests.json", "r") as file:
            data: dict = json.load(file)
    except json.decoder.JSONDecodeError:
        print("Error")
        sys.exit(1)

    for i in data:
        try:
            if (i['prompt'] == ""):
                print("Error : prompte is empty")
            prompt.append(i['prompt'])
        except (KeyError, json.decoder.JSONDecodeError):
            print("clee is empty")
            return
    return (prompt)
