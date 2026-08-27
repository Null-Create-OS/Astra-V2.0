import numpy as np
import tensorflow as tf

def top_p_sample(logits, temperature=0.7, top_p=0.9):
    logits = logits / max(temperature, 1e-5)
    probs = tf.nn.softmax(logits)
    values, indices = tf.math.top_k(probs, k=tf.minimum(50, tf.shape(probs)[-1]))
    sorted_probs = values / tf.reduce_sum(values)
    cumulative = tf.cumsum(sorted_probs)
    keep = cumulative <= top_p
    keep = tf.tensor_scatter_nd_update(
        keep, [[0]], [True]
    )
    filtered = tf.where(keep, sorted_probs, tf.zeros_like(sorted_probs))
    filtered /= tf.reduce_sum(filtered)
    choice = tf.random.categorical(tf.math.log([filtered]), 1)[0, 0]
    return int(indices[choice])

def generate(model, tokenizer, user_text, config):
    prompt = f"<user>{user_text}<assistant>"
    ids = tokenizer.encode(prompt)
    generated = []
    recent = []

    for _ in range(config["max_new_tokens"]):
        context = ids[-config["seq"]:]
        x = tf.constant([context], dtype=tf.int32)
        logits = model(x, training=False)[0, -1]
        for token_id in set(recent[-4:]):
            logits = tf.tensor_scatter_nd_update(
                logits, [[token_id]], [logits[token_id] - 1.0]
            )
        token_id = top_p_sample(logits, config["temp"], config["top_p"])

        if token_id == tokenizer.eos_id:
            break
        if token_id in {tokenizer.user_id, tokenizer.assistant_id}:
            break

        ids.append(token_id)
        generated.append(token_id)
        recent.append(token_id)

    return tokenizer.decode(generated).strip()

def chat(model, tokenizer, config):
    print("\n================================")
    print("          ASTRA V5 CHAT")
    print("================================\n")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        answer = generate(model, tokenizer, user, config)
        print(f"Astra: {answer}\n")
