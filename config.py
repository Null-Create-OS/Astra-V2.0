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
        "epochs": 50,
        "temp": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 50,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "shuffle_buffer": 5000,
    }
}
