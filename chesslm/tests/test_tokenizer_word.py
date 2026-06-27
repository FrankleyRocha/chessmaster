"""Testes para o tokenizador WordLevel com longest-match (ChessTokenizerWord)."""

import os
import tempfile

from model.tokenizer_word import ChessTokenizerWord


class TestChessTokenizerWord:
    def test_init(self):
        tok = ChessTokenizerWord()
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3
        assert tok.vocab_size == 211

    def test_train_is_noop(self):
        tok = ChessTokenizerWord()
        tok.train()
        tok.train("qualquer/caminho.txt")

    @property
    def vocab_size(self):
        return ChessTokenizerWord().vocab_size

    def test_piece_square_encode(self):
        tok = ChessTokenizerWord()
        assert tok.encode("Ba4") == [tok.stoi["B"], tok.stoi["a4"]]
        assert tok.encode("Nf3") == [tok.stoi["N"], tok.stoi["f3"]]
        assert tok.encode("Re1") == [tok.stoi["R"], tok.stoi["e1"]]

    def test_capture_check(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("Bxf7+")
        assert ids == [tok.stoi["B"], tok.stoi["x"], tok.stoi["f7"], tok.stoi["+"]]

    def test_promotion(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("e8=Q")
        assert ids == [tok.stoi["e8"], tok.stoi["="], tok.stoi["Q"]]

    def test_castling(self):
        tok = ChessTokenizerWord()
        assert tok.encode("O-O") == [tok.stoi["O-O"]]
        assert tok.encode("O-O-O") == [tok.stoi["O-O-O"]]

    def test_results(self):
        tok = ChessTokenizerWord()
        assert tok.encode("1-0") == [tok.stoi["1-0"]]
        assert tok.encode("0-1") == [tok.stoi["0-1"]]
        assert tok.encode("1/2-1/2") == [tok.stoi["1/2-1/2"]]
        assert tok.encode("*") == [tok.stoi["*"]]

    def test_move_number(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("1. e4")
        assert ids == [tok.stoi["1."], tok.stoi[" "], tok.stoi["e4"]]

    def test_move_number_high(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("99. d4")
        assert ids == [tok.stoi["99."], tok.stoi[" "], tok.stoi["d4"]]

    def test_disambiguation(self):
        tok = ChessTokenizerWord()
        # Rad1 = rook from a to d1
        ids = tok.encode("Rad1")
        assert ids == [tok.stoi["R"], tok.stoi["a"], tok.stoi["d1"]]
        # Nbd7 = knight from b to d7
        ids = tok.encode("Nbd7")
        assert ids == [tok.stoi["N"], tok.stoi["b"], tok.stoi["d7"]]

    def test_annotations(self):
        tok = ChessTokenizerWord()
        assert tok.encode("!!") == [tok.stoi["!!"]]
        assert tok.encode("!?") == [tok.stoi["!?"]]
        assert tok.encode("?!") == [tok.stoi["?!"]]
        assert tok.encode("!") == [tok.stoi["!"]]
        assert tok.encode("?") == [tok.stoi["?"]]

    def test_full_game(self):
        tok = ChessTokenizerWord()
        game = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O"
        ids = tok.encode(game)
        decoded = tok.decode(ids)
        assert decoded == game

    def test_encode_with_special_tokens(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("e4", add_special_tokens=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        assert ids[1:-1] == tok.encode("e4")

    def test_decode_skips_special(self):
        tok = ChessTokenizerWord()
        ids = [tok.bos_id, *tok.encode("e4"), tok.eos_id, tok.pad_id]
        assert tok.decode(ids) == "e4"

    def test_decode_keeps_special(self):
        tok = ChessTokenizerWord()
        ids = [tok.bos_id, *tok.encode("e4"), tok.eos_id]
        decoded = tok.decode(ids, skip_special=False)
        assert "<BOS>" in decoded
        assert "<EOS>" in decoded

    def test_unknown_chars(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("yz")
        assert all(i == tok.unk_id for i in ids)

    def test_partial_unknown(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("xyz")
        # 'x' IS in vocab (symbol), 'y' and 'z' are not
        assert ids[0] != tok.unk_id
        assert ids[1] == tok.unk_id
        assert ids[2] == tok.unk_id

    def test_empty_string(self):
        tok = ChessTokenizerWord()
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_ellipsis(self):
        tok = ChessTokenizerWord()
        ids = tok.encode("1... e5")
        assert ids == [tok.stoi["1."], tok.stoi["."], tok.stoi["."], tok.stoi[" "], tok.stoi["e5"]]

    def test_longest_match_priority(self):
        """Tokens maiores devem ter prioridade sobre menores."""
        tok = ChessTokenizerWord()
        # 'e5' (token casa) deve ser escolhido antes de 'e' + '5'
        ids = tok.encode("e5")
        # Se 'e5' for matched como um token só, len(ids) == 1
        assert len(ids) == 1
        assert ids[0] == tok.stoi["e5"]

    def test_save_load(self):
        tok = ChessTokenizerWord()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = ChessTokenizerWord.load(path)
            assert tok2.vocab_size == tok.vocab_size
            assert tok2.pad_id == tok.pad_id
            assert tok2.unk_id == tok.unk_id
            assert tok2.bos_id == tok.bos_id
            assert tok2.eos_id == tok.eos_id
            test = "1. e4 e5 2. Nf3 Nc6"
            assert tok2.encode(test) == tok.encode(test)
            assert tok2.decode(tok2.encode(test)) == test
        finally:
            os.unlink(path)

    def test_deterministic(self):
        tok = ChessTokenizerWord()
        text = "1. e4 e5 2. Nf3 Nc6 3. Bb5+ a6"
        assert tok.encode(text) == tok.encode(text)

    def test_repr(self):
        assert "ChessTokenizerWord" in repr(ChessTokenizerWord())

    def test_vocab_coverage(self):
        """Nenhum caractere desconhecido em PGN típico."""
        tok = ChessTokenizerWord()
        pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe5 18. Bxf6 Bxf6 19. Nbd2 Qc7 20. Bc2 Re8 1/2-1/2"
        ids = tok.encode(pgn)
        assert tok.unk_id not in ids
