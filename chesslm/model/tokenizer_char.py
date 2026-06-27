"""
Tokenizador por caractere para notação PGN de xadrez.

Vocabulário: ~50 símbolos que cobrem toda a notação algébrica padrão,
incluindo edge cases (xeque +, mate #, promoção =, roque O-O-O).
"""

import json
from pathlib import Path


# Vocabulário fixo baseado na notação PGN
CHESS_CHARS = (
    "abcdefgh"          # colunas
    "12345678"          # linhas
    "RNBQK"             # peças (maiúsculas)
    "x+=#+."            # captura, promoção, xeque, mate, ponto
    "O-"                # roque (letra O e hífen)
    "0123456789"        # dígitos para numeração dos movimentos
    " \n/"              # separadores
    "*"                 # resultado indeterminado
    "1/2"               # empate (caracteres já cobertos, mas explícito)
)

# Remove duplicatas mantendo ordem
_seen = set()
VOCAB_CHARS = []
for c in CHESS_CHARS:
    if c not in _seen:
        VOCAB_CHARS.append(c)
        _seen.add(c)

# Tokens especiais
PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
BOS_TOKEN  = "<BOS>"
EOS_TOKEN  = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class ChessTokenizerChar:
    """
    Tokenizador por caractere para partidas de xadrez em PGN.

    Uso:
        tok = ChessTokenizerChar()
        ids = tok.encode("1. e4 e5 2. Nf3")
        text = tok.decode(ids)
    """

    def __init__(self):
        # Constrói vocabulário: especiais primeiro, depois caracteres
        all_tokens = SPECIAL_TOKENS + VOCAB_CHARS
        self.stoi = {ch: i for i, ch in enumerate(all_tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

    def train(self, corpus_path: str | list[str] | None = None) -> None:
        """No-op: tokenizador char-level não precisa de treino."""

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Converte string PGN em lista de IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Converte lista de IDs de volta para string."""
        chars = []
        for i in ids:
            token = self.itos.get(i, "")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str):
        """Salva o vocabulário em JSON."""
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizador salvo em {path} (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> "ChessTokenizerChar":
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
        return tok

    def __repr__(self):
        return f"ChessTokenizerChar(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    tok = ChessTokenizerChar()
    print(tok)
    print(f"vocab_size : {tok.vocab_size}")
    print(f"vocabulário: {''.join(VOCAB_CHARS)}")

    test = "1. e4 1... e5 2. Nf3 Nc6 3. Bb5+ a6"
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)
    print(f"\noriginal : {test}")
    print(f"encoded  : {encoded}")
    print(f"decoded  : {decoded}")
    assert decoded == test, "Encode/decode não bateu!"
    print("✓ Encode/decode OK")

    tok.save("tokenizer_char.json")
