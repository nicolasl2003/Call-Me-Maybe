import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = "input"
OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function calling tool")
    parser.add_argument(
        "--function_definition",
        type=str,
        default="input/functions_definition.json",
        help="Path to the function definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input/function_calling_tests.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="output/function_calling_results.json"
    )
    return (parser.parse_args())
