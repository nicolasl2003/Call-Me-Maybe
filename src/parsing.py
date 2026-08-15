import json
from .generate_number import (generate_number,
                              generate_string
                              )
from pydantic import BaseModel, model_validator, Field, ValidationError
from llm_sdk import Small_LLM_Model
from typing import Literal
import sys


class Prompt(BaseModel):
    """Represent and validate a single natural-language prompt.

    This pydantic model ensures that a prompt provided in the input file
    is well-formed (non-empty) before being processed by the function
    calling pipeline.

    Attributes:
        valid_prompt: The natural-language prompt text, mapped from the
            "prompt" key in the input JSON.
    """
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
        """Parse and validate a list of raw prompt dictionaries.

        Each dictionary in `raw_data` is validated against the Prompt
        schema. Items that fail validation are skipped and logged, so
        that a single malformed entry does not interrupt the processing
        of the whole list.

        Args:
            raw_data: A list of dictionaries, each expected to contain
                a "prompt" key with a string value.

        Returns:
            A list of valid Prompt instances. Invalid entries are
            silently skipped (after printing an error message).
        """
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
    """Represent the type specification of a function parameter.

    This pydantic model validates that a parameter's declared type
    matches one of the JSON-schema-like types supported by the
    function calling pipeline.

    Attributes:
        type: The expected type of the parameter. Must be one of
            "number", "string", "boolean", "integer", "array", or
            "object".
    """
    type: Literal["number", "string", "boolean", "integer", "array", "object"]


class Definition(BaseModel):
    """Represent and validate a callable function's definition.

    This pydantic model describes a function that can be selected and
    called by the LLM, including its name, description, and expected
    parameters with their types.

    Attributes:
        name: The unique name of the function (e.g., "fn_add_numbers").
        description: A human-readable description of what the function
            does, used to help the LLM choose the correct function.
        parameters: A mapping of parameter names to their type
            specifications (ParamSpec).
    """
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


def select_function(model: Small_LLM_Model, token: str,
                    definitions: list[dict],
                    name: str,
                    parameter_schema: dict) -> list:
    """Generate parameter values for a selected function.

    Iterates over the parameter schema of a chosen function and
    generates a value for each parameter using the LLM, based on its
    declared type. For "number" and "integer" types, the generation
    returns early with a comma-split list. For "boolean" and "string"
    types, generated values are accumulated incrementally in
    `filled_param` (used as context for subsequent parameter
    generation) and appended to `result`.

    Args:
        model: The loaded small LLM model instance used for generation.
        token: The tokenizer associated with the model.
        definitions: The list of available function definitions,
            used as context for the LLM.
        name: The name of the selected function for which parameters
            are being generated.
        parameter_schema: A mapping of parameter names to their type
            specification (as dicts with a "type" key), describing
            the parameters expected by the function.

    Returns:
        A list of generated parameter values, in the order they were
        processed.
    """
    result = []
    filled_param: dict = {}
    for param_name, paran_info in parameter_schema.items():
        param_type = paran_info['type']
        if (param_type == "number"):
            value = generate_number(model, token, definitions, name)
            result = value.split(',')
            return (result)
        elif (param_type == "integer"):
            value = generate_number(model, token, definitions, name)
            result = value.split(',')
            return (result)
        elif (param_type == "boolean"):
            value = generate_string(model, token, definitions,
                                    name, param_name, filled_param)
            filled_param.update({param_name: value})
            result.append(value)
        elif (param_type == "string"):
            value = generate_string(model, token, definitions,
                                    name, param_name, filled_param)
            filled_param.update({param_name: value})
            result.append(value)
        else:
            print(f"Type not supporte : {param_type} for {param_name}")
            value = None
    return (result)


def load_json_file(path: str) -> list[dict]:
    """Load and parse a JSON file expected to contain a JSON array.

    Attempts to open and parse the file at `path` as JSON. Handles
    common file access errors and malformed JSON gracefully by
    printing an error message and exiting the program. If the parsed
    content is not a list, an error is printed and an empty list is
    returned instead of exiting.

    Args:
        path: The filesystem path to the JSON file to load.

    Returns:
        The parsed JSON content as a list of dictionaries. Returns an
        empty list if the parsed content is not a JSON array.

    Raises:
        SystemExit: If the file cannot be found or accessed
            (FileNotFoundError, IsADirectoryError, NotADirectoryError,
            PermissionError), or if the file content is not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError,
            PermissionError):
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
    """Load and parse prompt data from a JSON file into Prompt objects.

    Reads the JSON file at `path`, expected to contain an array of
    prompt objects, and converts each valid entry into a `Prompt`
    instance. Invalid entries are skipped (handled by
    `Prompt.pars_list`).

    Args:
        path: The filesystem path to the JSON file containing the
            prompt definitions.

    Returns:
        A list of validated `Prompt` instances parsed from the file.
    """
    raw_data: list[dict] = load_json_file(path)
    return (Prompt.pars_list(raw_data))


def parsing_definition(path: str) -> list[Definition]:
    """Load and parse function definition data from a JSON file.

    Reads the JSON file at `path`, expected to contain an array of
    function definition objects, and converts each valid entry into
    a `Definition` instance. Invalid entries are skipped (handled by
    `Definition.pars_list`).

    Args:
        path: The filesystem path to the JSON file containing the
            function definitions.

    Returns:
        A list of validated `Definition` instances parsed from the
        file.
    """
    raw_data = load_json_file(path)
    return (Definition.pars_list(raw_data))
