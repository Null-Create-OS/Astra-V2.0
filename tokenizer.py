import json
import re
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]

class AstraTokenizer:
    def __init__(self, vocab_size=128, min_frequency=1):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.token_to_id = {}
        self.id_to_token = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in SPECIAL_TOKENS:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def tokenize(self, text):
        pattern = r"<user>|<assistant>|<eos>|<bos>|<pad>|<unk>|[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\w\s]"
        return re.findall(pattern, text)

    def train(self, text):
        counts = Counter(self.tokenize(text))
        candidates = [
            (t, c) for t, c in counts.items()
            if c >= self.min_frequency and t not in SPECIAL_TOKENS
        ]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        for token, _ in candidates[:max(0, self.vocab_size - len(SPECIAL_TOKENS))]:
            self._add_token(token)
        return self

    def encode(self, text):
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(t, unk) for t in self.tokenize(text)]

    def decode(self, ids):
        tokens = [self.id_to_token.get(int(i), "<unk>") for i in ids]
        return self._detokenize(tokens)

    def _detokenize(self, tokens):
        out = ""
        no_space_before = {".", ",", "!", "?", ";", ":", "%", "'", "’"}
        for token in tokens:
            if token in {"<pad>", "<bos>", "<unk>"}:
                continue
            if token in {"<eos>", "<user>", "<assistant>"}:
                if out and not out.endswith(" "):
                    out += " "
                out += token
            elif not out:
                out = token
            elif token in no_space_before or out.endswith(("'", "’")):
                out += token
            else:
                out += " " + token
        return out.strip()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "token_to_id": self.token_to_id,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(data["vocab_size"], data["min_frequency"])
        obj.token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
        obj.id_to_token = {v: k for k, v in obj.token_to_id.items()}
        return obj

    @property
    def vocab_size_actual(self): return len(self.token_to_id)
    @property
    def pad_id(self): return self.token_to_id["<pad>"]
    @property
    def unk_id(self): return self.token_to_id["<unk>"]
    @property
    def bos_id(self): return self.token_to_id["<bos>"]
    @property
    def eos_id(self): return self.token_to_id["<eos>"]
    @property
    def user_id(self): return self.token_to_id["<user>"]
    @property
    def assistant_id(self): return self.token_to_id["<assistant>"]
