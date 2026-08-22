import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the function calling tool.

    Defines and parses the CLI arguments controlling the input/output
    file paths used by the function calling pipeline.

    Args:
        None.

    Returns:
        An `argparse.Namespace` containing the parsed arguments:
            - function_definition (str): Path to the JSON file
              containing function definitions.
            - input (str): Path to the JSON file containing the
              function calling test prompts.
            - output (Path): Path to the JSON file where the
              function calling results will be written.
    """
    parser = argparse.ArgumentParser(description="Function calling tool")
    parser.add_argument(
        "--function_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/output/function_calling_results.json"
    )
    return (parser.parse_args())
