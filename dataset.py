import re
import numpy as np
import tensorflow as tf

def parse_conversations(text):
    pattern = re.compile(
        r"<user>\s*(.*?)\s*<assistant>\s*(.*?)(?=<user>|\Z)",
        re.S | re.I,
    )
    pairs = []
    for user, assistant in pattern.findall(text):
        user, assistant = user.strip(), assistant.strip()
        if user and assistant:
            pairs.append((user, assistant))
    return pairs

def build_training_sequences(text, tokenizer, seq=64, stride=32):
    pairs = parse_conversations(text)
    examples = []
    for user, assistant in pairs:
        formatted = f"<user>{user}<assistant>{assistant}<eos>"
        ids = tokenizer.encode(formatted)
        if len(ids) < 2:
            continue
        for start in range(0, max(1, len(ids) - 1), stride):
            chunk = ids[start:start + seq + 1]
            if len(chunk) >= 2:
                chunk += [tokenizer.pad_id] * max(0, seq + 1 - len(chunk))
                examples.append(chunk[:seq + 1])
            if start + seq + 1 >= len(ids):
                break
    return np.asarray(examples, dtype=np.int32)

def make_dataset(sequences, batch_size, shuffle_buffer=5000):
    x = sequences[:, :-1]
    y = sequences[:, 1:]
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(min(len(x), shuffle_buffer), reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
