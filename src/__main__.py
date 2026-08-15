from .parsing import (
    parsing_prompt,
    select_function,
    parsing_definition
)
from .generate_json import generate_json
from .parse_args import parse_args
import json
import sys
from typing import Any
from .generate_function import generate_name
try:
    from llm_sdk import Small_LLM_Model
except KeyboardInterrupt:
    sys.exit()
from pathlib import Path


def main() -> None:
    """Run the function-calling pipeline from prompts to JSON output.

    This function orchestrates the full workflow:
        1. Parse CLI arguments (input file, function definitions file,
        output path).
        2. Parse the user prompts and the available function definitions.
        3. For each prompt, select the most appropriate function using the
           language model, based on the available function definitions.
        4. Extract the required parameters for the selected function from
           the prompt text.
        5. Generate a structured JSON object representing the function call
           (name and arguments) and append it to the result list.
        6. Write the accumulated results to the output file after each
           processed prompt.

    The function relies on external helpers:
        - `parse_args` to retrieve CLI arguments.
        - `parsing_prompt` to load and validate prompts.
        - `parsing_definition` to load and validate function definitions.
        - `generate_name` to select the function name via the model.
        - `select_function` to extract parameter values via the model.
        - `generate_json` to build the final JSON function call.

    Raises:
        SystemExit: If function definitions cannot be parsed, or if the
            output file cannot be created.

    Returns:
        None
    """
    model = Small_LLM_Model()
    args = parse_args()
    try:
        print(args.input)
        prompt = parsing_prompt(args.input)
    except ValueError as e:
        print(e)
    try:
        definitions = parsing_definition(args.function_definition)
    except ValueError as e:
        print(e)
        sys.exit(1)
    definitions_dict: list[dict[str, Any]] = [d.model_dump() for d in
                                              definitions]
    if not prompt:
        print("Error: no valid prompts to process")
        return
    if not definitions:
        print("Error: no valid function definitions available")
        return

    function_text = ""
    all_json: list[dict] = []
    valid_names = [d.name for d in definitions]
    for fn in definitions:
        function_text += f"- name : {fn.name} description : {fn.description}\n"
    print(f"\nValid names : {valid_names}")
    nb_prompt: int = 1
    for token in prompt:
        text = token.valid_prompt
        prompt_function = (
            "<|im_start|>system\n"
            "You are a function-calling assistant. "
            "Choose the SINGLE most appropriate function for"
            " the user's request.\n"
            "Available functions:\n"
            f"{function_text}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{token}<|im_end|>\n"
            "<|im_start|>assistant\n"
            '{"name": "'
        )
        tokens = model.encode(prompt_function).tolist()
        name = generate_name(tokens, valid_names, model)
        selected_def = next((d for d in definitions_dict if d['name'] == name),
                            None)
        if selected_def is None:
            print(f"Error: no matching definition for '{name}'")
            continue
        parameter_schema = selected_def['parameters']
        result = select_function(model, text, definitions_dict, name,
                                 parameter_schema)

        print(f"Number of prompts : {nb_prompt} / {len(prompt)}")
        print(result)
        print(text)
        print(name, end="\n\n")
        try:
            all_json.append(generate_json(text, name, parameter_schema,
                                          result))
        except Exception:
            print("Error: unknown type")
        path = Path(args.output)
        path.parent.mkdir(511, True, True)
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_json, f, indent=2, ensure_ascii=False)
        except FileNotFoundError:
            print("Error: A file cannot be created")
            sys.exit(1)
        nb_prompt += 1


if (__name__ == "__main__"):
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
