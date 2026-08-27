import tensorflow as tf

def diagnose(model, tokenizer):
    prompt = "<user>Hello!<assistant>"
    expected = "Hi!"

    prompt_ids = tokenizer.encode(prompt)
    expected_ids = tokenizer.encode(expected)

    print("\nAstra V5 diagnosis")
    print("----------------------------------")
    print("Prompt:", repr(prompt))
    print("Expected:", repr(expected))
    print("\nTokenized prompt:")

    for i, token_id in enumerate(prompt_ids):
        print(f"  {i}: {token_id:4d} {tokenizer.id_to_token.get(token_id)!r}")

    x = tf.constant([prompt_ids], dtype=tf.int32)
    logits = model(x, training=False)[0]

    correct = 0
    print("\nPredictions:")

    for i, expected_id in enumerate(expected_ids):
        position = len(prompt_ids) - 1 + i
        if position >= logits.shape[0]:
            break
        probs = tf.nn.softmax(logits[position])
        predicted_id = int(tf.argmax(probs))
        ep = float(probs[expected_id])
        pp = float(probs[predicted_id])
        expected_token = tokenizer.id_to_token.get(expected_id, "<unk>")
        predicted_token = tokenizer.id_to_token.get(predicted_id, "<unk>")
        ok = predicted_id == expected_id
        correct += int(ok)
        print(
            f"  {i} expected={expected_token!r:<10} "
            f"predicted={predicted_token!r:<10} "
            f"expected_prob={ep:.4f} predicted_prob={pp:.4f} "
            f"[{'OK' if ok else 'X'}]"
        )

    print("\n----------------------------------")
    print(f"Teacher-forcing accuracy: {correct}/{len(expected_ids)}")
