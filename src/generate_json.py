def generate_params_(parameters):


def generate_json(prompt, name_function, parameters):
    result: dict = {
        "prompt": f"{prompt}",
        "name": f"{name_function}",
    }
