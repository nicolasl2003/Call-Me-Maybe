import json
from pathlib import Path
from .generate_number import (generate_number,
                              generate_string
                              )
from pydantic import BaseModel, model_validator, Field, ValidationError
from typing import Literal
import sys

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"


class Prompt(BaseModel):
    valid_prompt: str = Field(alias="prompt")

    @model_validator(mode="after")
    def check_prompt(self) -> "Prompt":
        if (self.valid_prompt == ""):
            raise ValueError(
                "Error: prompt is empty"
            )
        return (self)

    @classmethod
    def pars_list(cls, raw_data: list[dict]) -> list["Prompt"]:
        valid_prompts: list["Prompt"] = []
        for item in raw_data:
            try:
                p = cls.model_validate(item)
            except ValidationError as e:
                print(f"Error skipping item {item}: {e}")
                continue
            valid_prompts.append(p)
        return (valid_prompts)


class ParamSpec(BaseModel):
    type: Literal["number", "string", "boolean", "integer", "array", "object"]


class ReturnSpec(BaseModel):
    type: Literal["nomber", "string", "boolean", "integer", "array", "object"]


class Definition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamSpec]

    @model_validator(mode="after")
    def check_definition(self) -> "Definition":
        if (self.name == ""):
            raise ValueError("Error: function is empty")
        if (self.description == ""):
            raise ValueError("Error: description is empty")
        return (self)

    @classmethod
    def pars_list(cls, raw_data: list[dict]) -> list["Definition"]:
        valid_defs: list["Definition"] = []
        for item in raw_data:
            try:
                d = cls.model_validate(item)
            except Exception as e:
                print(f"Error: skipping definition {item}: {e}")
                sys.exit(1)
            valid_defs.append(d)
        return (valid_defs)


def select_function(model, token, definitions, name, parameter_schema):
    result = []

    for param_name, paran_info in parameter_schema.items():
        param_type = paran_info['type']
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


def load_json_file(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}")
        sys.exit(1)
    if (not isinstance(data, list)):
        print(f"Error: expected a JSON array in {path}")
        return []
    return (data)


def parsing_prompt(path: str) -> list[Prompt]:

    raw_data: list[dict] = load_json_file(path)
    return (Prompt.pars_list(raw_data))


def parsing_definition(path: str) -> list[Definition]:
    raw_data = load_json_file(path)
    return (Definition.pars_list(raw_data))
