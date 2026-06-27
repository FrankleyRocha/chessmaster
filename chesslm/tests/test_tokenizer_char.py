"""Testes para o tokenizador character-level (ChessTokenizerChar)."""

import os
import tempfile

from model.tokenizer_char import ChessTokenizerChar


class TestChessTokenizerChar:
    def test_init(self):
        tok = ChessTokenizerChar()
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3
        assert tok.vocab_size == 38  # 4 especiais + 34 chars únicos

    def test_train_is_noop(self):
        tok = ChessTokenizerChar()
        tok.train()      # não deve lançar erro
        tok.train("qualquer/caminho.txt")  # também não

    def test_encode_decode_roundtrip(self):
        tok = ChessTokenizerChar()
        test = "1. e4 e5 2. Nf3 Nc6 3. Bb5+ a6"
        ids = tok.encode(test)
        decoded = tok.decode(ids)
        assert decoded == test

    def test_encode_with_special_tokens(self):
        tok = ChessTokenizerChar()
        ids = tok.encode("e4", add_special_tokens=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        assert ids[1:-1] == tok.encode("e4")

    def test_decode_skips_special(self):
        tok = ChessTokenizerChar()
        ids = [tok.bos_id, *tok.encode("e4"), tok.eos_id, tok.pad_id]
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == "e4"

    def test_decode_keeps_special(self):
        tok = ChessTokenizerChar()
        ids = [tok.bos_id, *tok.encode("e4"), tok.eos_id]
        decoded = tok.decode(ids, skip_special=False)
        assert "<BOS>" in decoded

    def test_unknown_chars_mapped_to_unk(self):
        tok = ChessTokenizerChar()
        # 'y' e 'z' não estão no vocabulário PGN
        ids = tok.encode("yz")
        assert all(i == tok.unk_id for i in ids)

    def test_empty_string(self):
        tok = ChessTokenizerChar()
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_castling(self):
        tok = ChessTokenizerChar()
        for castle in ["O-O", "O-O-O"]:
            assert tok.decode(tok.encode(castle)) == castle

    def test_check_and_mate(self):
        tok = ChessTokenizerChar()
        for move in ["Bb5+", "Qh7#"]:
            assert tok.decode(tok.encode(move)) == move

    def test_save_load(self):
        tok = ChessTokenizerChar()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = ChessTokenizerChar.load(path)
            assert tok2.vocab_size == tok.vocab_size
            assert tok2.pad_id == tok.pad_id
            assert tok2.unk_id == tok.unk_id
            assert tok2.bos_id == tok.bos_id
            assert tok2.eos_id == tok.eos_id
            assert tok2.encode("e4") == tok.encode("e4")
        finally:
            os.unlink(path)

    def test_repr(self):
        assert "ChessTokenizerChar" in repr(ChessTokenizerChar())

    def test_vocabulary_coverage(self):
        """Verifica se cobre todos os símbolos básicos do PGN."""
        tok = ChessTokenizerChar()
        pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O * 1/2-1/2"
        ids = tok.encode(pgn)
        assert tok.unk_id not in ids  # nenhum caractere desconhecido
