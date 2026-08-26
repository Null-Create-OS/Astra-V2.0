import os
import argparse

import tensorflow as tf

from model import Astra
from tokenizer import AstraTokenizer

from train import train
from chat import chat
from diagnose import diagnose


CONFIGS = {

    "tiny": {
        "vocab_size": 128,
        "min_frequency": 1,

        "d": 128,
        "layers": 3,
        "heads": 4,
        "ff": 256,

        "seq": 64,
        "stride": 32,

        "batch": 16,

        "lr": 0.0003,
        "epochs": 30,

        "temp": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 80,

        "dropout": 0.1,
        "weight_decay": 0.01,
        "shuffle_buffer": 5000,
    },

    "small": {

        "vocab_size": 2000,
        "min_frequency": 1,

        "seq": 256,
        "d": 384,
        "layers": 8,
        "heads": 8,
        "ff": 1536,

        "batch": 2,

        "lr": 0.0002,
        "epochs": 5,

        "temp": 0.7,
        "top_p": 0.9,

        "max_new_tokens": 120,

        "dropout": 0.1,
        "weight_decay": 0.01,

        "shuffle_buffer": 10000,
    }
}


def create_model(
    config,
    vocab_size
):

    model = Astra(
        vocab_size=vocab_size,
        seq_len=config["seq"],
        d_model=config["d"],
        layers=config["layers"],
        heads=config["heads"],
        ff_dim=config["ff"],
        dropout=config.get(
            "dropout",
            0.1
        )
    )

    dummy = tf.zeros(
        (1, config["seq"]),
        dtype=tf.int32
    )

    model(
        dummy,
        training=False
    )

    return model


def find_checkpoint(
    config_name
):

    directory = os.path.join(
        "checkpoints",
        "v5"
    )

    if not os.path.exists(
        directory
    ):
        return None

    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(),
        directory,
        max_to_keep=5
    )

    return manager.latest_checkpoint


def load_runtime(
    config_name
):

    config = CONFIGS[
        config_name
    ]

    tokenizer_path = (
        "tokenizer.json"
    )

    if not os.path.exists(
        tokenizer_path
    ):
        raise RuntimeError(
            "tokenizer.json does not exist.\n"
            "Run training first."
        )

    tokenizer = AstraTokenizer.load(
        tokenizer_path
    )

    model = create_model(
        config,
        tokenizer.vocab_size
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config["lr"],
        weight_decay=config.get(
            "weight_decay",
            0.01
        )
    )

    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer
    )

    checkpoint_dir = os.path.join(
        "checkpoints",
        "v5"
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        checkpoint_dir,
        max_to_keep=5
    )

    checkpoint_path = (
        manager.latest_checkpoint
    )

    if checkpoint_path:

        print()
        print(
            "Loading checkpoint:",
            checkpoint_path
        )

        try:

            checkpoint.restore(
                checkpoint_path
            ).expect_partial()

            print(
                "Checkpoint loaded."
            )

        except Exception as e:

            print(
                "Could not load checkpoint:"
            )

            print(e)

    else:

        print()
        print(
            "No V5 checkpoint found."
        )

        print(
            "Astra is running with fresh weights."
        )

    return (
        config,
        tokenizer,
        model
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="tiny",
        choices=CONFIGS.keys()
    )

    parser.add_argument(
        "--mode",
        default="chat",
        choices=[
            "train",
            "chat",
            "diagnose"
        ]
    )

    args = parser.parse_args()

    config = CONFIGS[
        args.config
    ]

    print(
        "TensorFlow:",
        tf.__version__,
        " GPUs:",
        len(
            tf.config.list_physical_devices(
                "GPU"
            )
        )
    )

    print(
        "config:",
        config
    )

    if args.mode == "train":

        train(
            config=config,
            data_dir="data",
            tokenizer_path="tokenizer.json",
            checkpoint_dir="checkpoints/v5"
        )

        return

    (
        config,
        tokenizer,
        model
    ) = load_runtime(
        args.config
    )

    print(
        "vocabulary:",
        tokenizer.vocab_size
    )

    print(
        "parameters:",
        model.count_params()
    )

    if args.mode == "diagnose":

        diagnose(
            model,
            tokenizer,
            config
        )

    elif args.mode == "chat":

        chat(
            model,
            tokenizer,
            config
        )


if __name__ == "__main__":
    main()