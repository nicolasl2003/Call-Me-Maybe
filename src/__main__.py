from .parsing import (
    parsing_prompt,
    generate_name,
    select_function,
    parsing_definition
)

from .generate_json import generate_json
from llm_sdk import Small_LLM_Model


def main():
    model = Small_LLM_Model()
    prompt = parsing_prompt()
    function_text = ""
    definitions = parsing_definition()
    for fn in definitions:
        function_text += f"- name : {fn['name']} description : {fn['description']}\n"
        valid_names = [d["name"] for d in definitions]
        print(f"\nNoms valides : {valid_names}")
    for token in prompt:
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
        selected_def = next((d for d in definitions if d["name"] == name),
                            None)
        parameter_schema = selected_def["parameters"]
        result = select_function(model, token, definitions, name,
                                 parameter_schema)
        all_json = []

        all_json.append(generate_json(token, name, definitions, result))
        print(f"parametre : {result}")
        print(f"name : {name}")
        with open("data/output/function_calling_results.json", "w" as f):
            json.dump(all_json; f; indent=2)



if (__name__ == "__main__"):
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
