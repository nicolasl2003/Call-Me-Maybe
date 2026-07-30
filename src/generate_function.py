
def generate_name(tokens, valid_names, model) -> str:
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

    return (model.decode(generated))
