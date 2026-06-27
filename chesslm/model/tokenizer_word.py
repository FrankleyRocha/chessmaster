"""
Tokenizador WordLevel com longest-match para notação PGN de xadrez.

Vocabulário fixo de ~210 tokens (casas, peças, símbolos, etc.).
Tokenização determinística: sempre escolhe o maior token que casa.
"""

import json
from pathlib import Path

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

FILES = list("abcdefgh")
RANKS = list("12345678")
PIECES = ["K", "Q", "R", "B", "N"]
DIGITS = [str(i) for i in range(10)]
SYMBOLS = ["x", "=", "+", "#", ".", "-", "/"]
ANNOTATIONS = ["!", "?", "!!", "??", "!?", "?!"]
CASTLE = ["O-O", "O-O-O"]
RESULTS = ["1-0", "0-1", "1/2-1/2", "*"]
SEPARATORS = [" ", "\n"]


def _build_vocab():
    tokens = list(SPECIAL_TOKENS)
    tokens.extend(SEPARATORS)
    tokens.extend(PIECES)
    tokens.extend(FILES)
    tokens.extend(DIGITS)
    for f in FILES:
        for r in RANKS:
            tokens.append(f + r)
    for i in range(1, 100):
        tokens.append(f"{i}.")
    tokens.extend(RESULTS)
    tokens.extend(SYMBOLS)
    tokens.extend(CASTLE)
    tokens.extend(ANNOTATIONS)
    return tokens


class ChessTokenizerWord:
    """
    Tokenizador WordLevel com longest-match para PGN de xadrez.

    Uso:
        tok = ChessTokenizerWord()
        ids = tok.encode("1. e4 e5 2. Nf3")
        text = tok.decode(ids)
    """

    def __init__(self):
        all_tokens = _build_vocab()
        self.stoi = {ch: i for i, ch in enumerate(all_tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

        self._special_set = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}

        # Pré-computa tokens agrupados por tamanho (decrescente) para longest-match
        self._lengths = sorted({len(t) for t in all_tokens}, reverse=True)

    def train(self, corpus_path: str | list[str] | None = None) -> None:
        """No-op: tokenizador word-level não precisa de treino."""

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Converte string PGN em lista de IDs (longest-match)."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)

        i = 0
        n = len(text)
        while i < n:
            matched = False
            for length in self._lengths:
                if i + length <= n:
                    chunk = text[i:i + length]
                    token_id = self.stoi.get(chunk)
                    if token_id is not None:
                        ids.append(token_id)
                        i += length
                        matched = True
                        break
            if not matched:
                ids.append(self.unk_id)
                i += 1

        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Converte lista de IDs de volta para string."""
        chars = []
        for i in ids:
            token = self.itos.get(i, "")
            if skip_special and token in self._special_set:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str) -> None:
        """Salva o vocabulário em JSON."""
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizador salvo em {path} (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> "ChessTokenizerWord":
        """Carrega tokenizador de um arquivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.pad_id = tok.stoi[PAD_TOKEN]
        tok.unk_id = tok.stoi[UNK_TOKEN]
        tok.bos_id = tok.stoi[BOS_TOKEN]
        tok.eos_id = tok.stoi[EOS_TOKEN]
        tok._special_set = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        tok._lengths = sorted({len(t) for t in tok.stoi}, reverse=True)
        return tok

    def __repr__(self) -> str:
        return f"ChessTokenizerWord(vocab_size={self.vocab_size})"
