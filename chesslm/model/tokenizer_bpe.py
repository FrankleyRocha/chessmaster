"""
Tokenizador BPE (Byte-Pair Encoding) para notação PGN de xadrez.

Treina um vocabulário subword no corpus de partidas,
aprendendo padrões comuns (e4, Nf3, O-O) como tokens únicos.
"""

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class ChessTokenizerBPE:
    """
    Tokenizador BPE para partidas de xadrez em PGN.

    Uso:
        tok = ChessTokenizerBPE(vocab_size=512)
        tok.train("data/pretrain.txt")
        ids = tok.encode("1. e4 e5 2. Nf3")
        text = tok.decode(ids)
    """

    def __init__(self, vocab_size: int = 512):
        self._vocab_size = vocab_size

        self._tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()

        # IDs dos tokens especiais (setados após train/load)
        self.pad_id = None
        self.unk_id = None
        self.bos_id = None
        self.eos_id = None

    def train(self, corpus_path: str | list[str]) -> None:
        """Treina o BPE em um ou mais arquivos de texto PGN."""
        files = [corpus_path] if isinstance(corpus_path, str) else corpus_path

        trainer = trainers.BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        self._tokenizer.train(files, trainer)
        self._set_special_ids()

    def _set_special_ids(self) -> None:
        self.pad_id = self._tokenizer.token_to_id(PAD_TOKEN)
        self.unk_id = self._tokenizer.token_to_id(UNK_TOKEN)
        self.bos_id = self._tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = self._tokenizer.token_to_id(EOS_TOKEN)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Converte string PGN em lista de IDs."""
        ids = self._tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Converte lista de IDs de volta para string."""
        if skip_special:
            skip = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in skip]
        return self._tokenizer.decode(ids)

    def save(self, path: str) -> None:
        """Salva o tokenizador no formato HuggingFace tokenizers."""
        self._tokenizer.save(path)
        print(f"Tokenizador BPE salvo em {path} (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> "ChessTokenizerBPE":
        """Carrega tokenizador de um arquivo JSON (formato HuggingFace)."""
        tok = cls.__new__(cls)
        tok._tokenizer = Tokenizer.from_file(path)
        tok._set_special_ids()
        return tok

    def __repr__(self) -> str:
        return f"ChessTokenizerBPE(vocab_size={self.vocab_size})"
