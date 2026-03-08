import re

SPECIAL_TOKENS = [
    "<system>",
    "</system>",
    "<human>",
    "</human>",
    "<bot>",
    "</bot>",
    "<eos>",
]

class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def build(self, text):
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

        for word in self._split(text):
            if word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

        self.vocab_size = len(self.stoi)

    def _split(self, text):
        return re.findall(r"<[^>]+>|[\w']+|[^\w\s]", text.lower())

    def encode(self, text):
        return [self.stoi[t] for t in self._split(text) if t in self.stoi]

    def decode(self, ids):
        words = [self.itos.get(i, "") for i in ids]
        out = ""
        for w in words:
            if w in {".", ",", "!", "?", ":", ";", "'"}:
                out = out.rstrip(" ") + w + " "
            elif w in SPECIAL_TOKENS:
                out += w + " "
            else:
                out += w + " "
        return out.strip()

    def token_id(self, token):
        return self.stoi.get(token, -1)