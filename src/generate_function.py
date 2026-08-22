import sys

try:
    from llm_sdk import Small_LLM_Model
except KeyboardInterrupt:
    sys.exit()


def generate_name(tokens: list, valid_names: list[str],
                  model: Small_LLM_Model) -> str:
    """Generate the name of the function selected by the model.

    Performs constrained greedy decoding restricted to the set of
    valid function names: at each step, only tokens that keep the
    generated sequence a prefix of at least one encoded name in
    `valid_names` are allowed, guaranteeing the final output exactly
    matches one of the candidates.

    Args:
        tokens: The encoded input prompt (as produced by
            `model.encode`), used as the initial context for
            decoding.
        valid_names: The list of allowed function names the model
            may choose from.
        model: The `Small_LLM_Model` instance used for encoding,
            decoding and computing logits.

    Returns:
        The name (as a string) of the function chosen by the model,
        guaranteed to be one of `valid_names`.
    """
    encoded_names = {}
    generated: list = []

    for name in valid_names:
        encoded_names[name] = model.encode(name).tolist()[0]
    current_input = list(tokens[0])
    while (True):
        candidates = [
            enc for enc in encoded_names.values()
            if (enc[:len(generated)] == generated)
        ]
        allowed = set()
        for enc in candidates:
            if (len(enc) > len(generated)):
                allowed.add(enc[len(generated)])
        if (not allowed):
            break
        logits = model.get_logits_from_input_ids(current_input)
        best = max(allowed, key=lambda tok_id: logits[tok_id])
        generated.append(best)
        current_input.append(best)
    result: str = model.decode(generated)
    return (result)
