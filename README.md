*This project has been created as part of the 42 curriculum by <nilinott>.*

# call-me-maybe

## Description

`call-me-maybe` translates natural language requests into structured function
calls. Given a prompt such as `"What is the sum of 40 and 2?"`, the program does
not answer `42` — it returns the function to call and its arguments:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The whole pipeline runs on **Qwen/Qwen3-0.6B**, a 0.6B parameter model that is
notoriously unreliable at spontaneously producing valid JSON. Reliability is
obtained through **constrained decoding**: the JSON skeleton is written by the
program itself, and the model is only ever asked to produce the values, one
token at a time, under structural supervision.

The program reads two input files, processes every prompt, and writes a single
JSON array to `data/output/function_calling_results.json`.

## Instructions

### Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- The `llm_sdk` package, copied at the root of the project (next to `src/`)

### Installation

```bash
uv sync
```

### Running

```bash
uv run python -m src
```

With explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

| Target | Effect |
|---|---|
| `make run` | Run the program with default paths |
| `make debug` | Run under `pdb` |
| `make lint` | `flake8 .` and `mypy .` with the required flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, caches |

## Algorithm explanation

The program never asks the model for a complete JSON object. Generation is split
into two independent stages, and the structure is always produced by the code.

### Stage 1 — Function selection

The model is shown the list of available function names and a set of few-shot
examples built from the definitions file. Decoding is constrained so that only
tokens leading to a valid function name can be emitted: at each step, the set of
candidate names still compatible with the tokens generated so far is computed,
and every logit outside that set is masked to `-inf`.

Because the candidate set shrinks with each token, the model is structurally
incapable of inventing a name that does not exist in
`functions_definition.json`. Selection is done by the LLM — no keyword matching,
no heuristic on the prompt.

### Stage 2 — Argument generation by context injection

Once the function is known, its parameters are read from the definitions file
and generated **one at a time**. For each parameter, the program builds a prefix
that already contains the valid JSON up to the exact point where the value must
appear:

```
{"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "Hello World", "regex": "
```

The model only has to continue from there. Two consequences:

- the key name is in immediate context, right before generation, which is far
  more directive than instructions placed earlier in the prompt;
- values already produced are re-injected, properly escaped, so each parameter
  is generated with full knowledge of the previous ones.

Generation stops on a closing quote, a newline, a brace, or a per-type length
budget. The value is then cast according to the declared type (`number`,
`string`, `boolean`) by a dedicated function, separate from generation.

The final object is assembled by the program. The JSON is serialised with
`json.dump`, never by the model.

## Design decisions

**Separation of generation and casting.** `generate_string` produces raw text
and nothing else. Type conversion lives in its own function, driven by the
`type` field of the definitions. Adding a new type means adding a branch in the
dispatcher, not touching the decoding loop.

**Structure written by the code, values by the model.** Braces, quotes, colons
and commas never come from the LLM. This is what guarantees 100% parseable
output regardless of model behaviour.

**Schema read at runtime.** Parameter names, order and types come from
`functions_definition.json`. Nothing about the example functions is hardcoded,
so a different definitions file works without modification.

**Pydantic for validation.** Input files and the final result are validated
against Pydantic models. A malformed definition is reported clearly instead of
propagating into generation.

**No `dspy`, `transformers`, `outlines` or equivalent.** Only `numpy`,
`pydantic`, `json` and the provided SDK. No private attribute of `llm_sdk` is
accessed.

## Performance analysis

Measured on a 79-prompt test file covering five functions.

| Metric | Result |
|---|---|
| Valid JSON | 100% |
| Correct function selection | 100% (79/79) |
| Exact parameter extraction | ~83% (66/79) |
| Total runtime | under the 5-minute budget |

Per-function breakdown:

| Function | Correct |
|---|---|
| `fn_add_numbers` | 15/15 |
| `fn_reverse_string` | 15/15 |
| `fn_greet` | 15/15 (3 with altered casing) |
| `fn_get_square_root` | 10/15 |
| `fn_substitute_string_with_regex` | 11/15 on the `regex` field |

Constrained decoding removes structural failure entirely. Remaining errors are
semantic — the model extracts the wrong value, never an invalid one. Runtime is
dominated by the number of forward passes, which is proportional to the total
number of generated tokens; short length budgets per parameter keep it low.

## Challenges faced

**The model solving instead of extracting.** On `"What is the square root of
25?"`, the model returned `5.0` rather than `25.0`. It happened specifically on
perfect squares it recognises, and never on values like 289 or 441. The prompt
asked for values that were *exact* but never explicitly forbade computing. Adding
`"Never compute, never solve. If the request says 'square root of 25', the
argument is 25, not 5"` to the header addressed it. Instructive failure mode: the
model was being helpful, not wrong.

**Anchoring on the first few-shot example.** The `regex` parameter fell back to
`\d+` on requests like `"Substitute the word 'dog' with 'cat'"`, where the
expected pattern is simply `dog`. The `digits -> \d+` mapping was the first
example listed and acted as an attractor. Reordering the examples and putting
the literal-word case first reduced the effect.

**Length budget colliding with a legitimate value.** The truncation budget for
`regex` was set to 12 characters — exactly the length of `[aeiouAEIOU]`. Values
were silently cut when the pattern was tokenised across several pieces. Raised
to a safer margin.

**Partial values leaking between parameters.** Before context injection was
introduced, generating `regex` sometimes pulled fragments of `source_string`
into the output. Re-injecting the already-filled parameters as valid escaped
JSON, rather than reusing a free-form prompt, eliminated it.

## Testing strategy

- **Structural validation**: every produced file is re-parsed with
  `json.load`, then checked against the schema — key presence, exact types, no
  extra keys.
- **Regression file**: a reference set of prompts with expected outputs, diffed
  after each prompt change to catch silent degradation on one function while
  fixing another.
- **Edge cases**: empty input file, missing file, invalid JSON, unknown function
  in the definitions, empty string arguments, negative and decimal numbers,
  apostrophes and quotes inside string values.
- **Error handling**: each failure mode is triggered manually to confirm the
  program exits with a clear message and never a traceback.

Tests are not part of the graded submission.

## Example usage

Given `data/input/function_calling_tests.json`:

```json
[
  {"prompt": "Reverse the string 'hello'"},
  {"prompt": "Replace all vowels in 'Data Science' with dash"}
]
```

Running:

```bash
uv run python -m src
```

Produces `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  },
  {
    "prompt": "Replace all vowels in 'Data Science' with dash",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Data Science",
      "regex": "[aeiouAEIOU]",
      "replacement": "dash"
    }
  }
]
```

## Resources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Documentation Open](https://python.pages.doc.ic.ac.uk/java/lessons/java/10-files/03-load.html#:~:text=To%20load%20your%20data%20from,load(file_object)%20.&text=To%20load%20your%20object%20directly,for%20'load%20string')

### Use of AI

AI assistants were used for:

- exploring approaches to constrained decoding before settling on the
  logit-masking implementation;
- iterating on prompt wording, especially the instructions that stopped the
  model from computing square roots instead of extracting them;
- reviewing type annotations for `mypy --strict` compliance.

All decoding logic, the generation pipeline and the schema handling were written
and debugged manually. Every AI suggestion was tested and rewritten before being
kept.