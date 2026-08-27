import tensorflow as tf


def make_sequences(
    tokens,
    sequence_length,
    stride,
):
    tokens = list(tokens)

    sequence_length = int(sequence_length)
    stride = int(stride)

    if sequence_length < 2:
        raise ValueError(
            "sequence_length must be at least 2."
        )

    if stride < 1:
        raise ValueError(
            "stride must be at least 1."
        )

    sequences = []

    # Need sequence_length input tokens
    # plus one token for the target.
    for start in range(
        0,
        len(tokens) - sequence_length,
        stride,
    ):
        inputs = tokens[
            start:start + sequence_length
        ]

        targets = tokens[
            start + 1:start + sequence_length + 1
        ]

        if len(inputs) != sequence_length:
            continue

        if len(targets) != sequence_length:
            continue

        sequences.append(
            (
                inputs,
                targets,
            )
        )

    return sequences


def make_dataset(
    sequences,
    batch_size,
    shuffle_buffer=5000,
):
    if not sequences:
        raise ValueError(
            "Cannot create a dataset from zero sequences."
        )

    inputs = []
    targets = []

    for inputs_seq, targets_seq in sequences:
        inputs.append(inputs_seq)
        targets.append(targets_seq)

    inputs = tf.convert_to_tensor(
        inputs,
        dtype=tf.int32,
    )

    targets = tf.convert_to_tensor(
        targets,
        dtype=tf.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            inputs,
            targets,
        )
    )

    dataset = dataset.shuffle(
        min(
            int(shuffle_buffer),
            len(inputs),
        ),
        reshuffle_each_iteration=True,
    )

    dataset = dataset.batch(
        int(batch_size),
        drop_remainder=False,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset