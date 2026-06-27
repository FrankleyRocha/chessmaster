"""Testes para a factory de tokenizadores (model/tokenizer.py)."""

import os
import tempfile

import pytest

from model.tokenizer import create_tokenizer, load_tokenizer
from model.tokenizer_char import ChessTokenizerChar
from model.tokenizer_bpe import ChessTokenizerBPE

SAMPLE_GAME = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O\n"


@pytest.fixture
def corpus_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_GAME * 50)
        path = f.name
    yield path
    os.unlink(path)


class TestCreateTokenizer:
    def test_create_bpe(self):
        tok = create_tokenizer("bpe")
        assert isinstance(tok, ChessTokenizerBPE)

    def test_create_bpe_with_vocab_size(self):
        tok = create_tokenizer("bpe", vocab_size=128)
        assert isinstance(tok, ChessTokenizerBPE)
        assert tok._vocab_size == 128

    def test_create_char(self):
        tok = create_tokenizer("char")
        assert isinstance(tok, ChessTokenizerChar)

    def test_create_invalid_type(self):
        with pytest.raises(ValueError, match="desconhecido"):
            create_tokenizer("wordpiece")


class TestLoadTokenizer:
    def test_load_bpe(self, corpus_file):
        tok_orig = create_tokenizer("bpe")
        tok_orig.train(corpus_file)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok_orig.save(path)
            tok_loaded = load_tokenizer(path, "bpe")
            assert isinstance(tok_loaded, ChessTokenizerBPE)
            assert tok_loaded.vocab_size == tok_orig.vocab_size
            assert tok_loaded.encode("e4") == tok_orig.encode("e4")
        finally:
            os.unlink(path)

    def test_load_char(self):
        tok_orig = create_tokenizer("char")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok_orig.save(path)
            tok_loaded = load_tokenizer(path, "char")
            assert isinstance(tok_loaded, ChessTokenizerChar)
            assert tok_loaded.vocab_size == tok_orig.vocab_size
        finally:
            os.unlink(path)

    def test_load_invalid_type(self):
        with pytest.raises(ValueError, match="desconhecido"):
            load_tokenizer("path.json", "wordpiece")


class TestIntegrationWithConfig:
    def test_create_from_config_type_bpe(self):
        from model.config import ModelConfig
        cfg = ModelConfig(tokenizer_type="bpe")
        tok = create_tokenizer(cfg.tokenizer_type)
        assert isinstance(tok, ChessTokenizerBPE)

    def test_create_from_config_type_char(self):
        from model.config import ModelConfig
        cfg = ModelConfig(tokenizer_type="char")
        tok = create_tokenizer(cfg.tokenizer_type)
        assert isinstance(tok, ChessTokenizerChar)

    def test_default_tokenizer_in_config(self):
        from model.config import ModelConfig
        cfg = ModelConfig()
        assert cfg.tokenizer_type == "bpe"
