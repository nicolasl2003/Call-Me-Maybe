from .parsing import (
    parsing_prompt,
    select_function,
    parsing_definition
)
from .generate_json import generate_json
from .parse_args import parse_args
import json
import sys
from .generate_function import generate_name
from llm_sdk import Small_LLM_Model


def main():
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
    definitions_dict = [d.model_dump() for d in definitions]
    if not prompt:
        print("Error: no valid prompts to process")
        return
    if not definitions:
        print("Error: no valid function definitions available")
        return

    function_text = ""
    all_json: list = []
    valid_names = [d.name for d in definitions]
    for fn in definitions:
        function_text += f"- name : {fn.name} description : {fn.description}\n"
    print(f"\nNoms valides : {valid_names}")

    for token in prompt:
        text = token.valid_prompt
        prompt_function = (
            "<|im_start|>system\n"
            "You are a function-calling assistant. "
            "Choose the SINGLE most appropriate function for the user's request.\n"
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

        print(result)
        print(text)
        print(name, end="\n\n")
        all_json.append(generate_json(text, name, parameter_schema, result))

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_json, f, indent=2, ensure_ascii=False)


if (__name__ == "__main__"):
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
