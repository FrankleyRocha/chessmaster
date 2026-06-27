"""Testes para o tokenizador BPE."""

import os
import tempfile
import pytest

from model.tokenizer_bpe import ChessTokenizerBPE

# Amostra de PGN para testes
SAMPLE_GAMES = """1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 11. O-O Nxc3 12. Rxc3 e5
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 Nh5 13. f4 exf4 14. Bxf4 Nc5 15. Qd4 Nxb3+ 16. axb3 Bxb3
1. Nf3 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. e4 Nxc3 6. dxc3 Bg7 7. Bc4 O-O 8. O-O c5 9. Qe2 Nc6 10. Rd1 Qc7 11. Bf4 Rd8 12. Rac1 b6 13. h3 Bb7
1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 7. Bb3 Ne7 8. h3 Ng6 9. Re1 Kh8 10. Nbd2 a5 11. a4 c6 12. Nf1 Be6 13. Ng3 Qd7
1. d4 Nf6 2. c4 e6 3. Nf3 b6 4. g3 Bb7 5. Bg2 Be7 6. O-O O-O 7. Nc3 Ne4 8. Qc2 Nxc3 9. Qxc3 f5 10. b3 Bf6 11. Bb2 d6 12. Rad1 Qe7 13. e3 a5
1. e4 c5 2. Nf3 e6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 Nc6 6. Be3 Bb4 7. Bd3 Nxd4 8. Bxd4 e5 9. Be3 d6 10. O-O O-O 11. Qd2 Be6 12. Nd5 g6 13. Rad1 Bxd5 14. exd5 Qd7
1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Nxe4 5. Re1 Nd6 6. Nxe5 Be7 7. Bf1 Nxe5 8. Rxe5 O-O 9. d4 Bf6 10. Re1 Re8 11. c3 Rxe1 12. Qxe1 Ne8 13. Bf4 d6 14. Nd2 Be6
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d5 9. exd5 Nxd5 10. Nxe5 Nxe5 11. Rxe5 c6 12. d4 Bd6 13. Re1 Qh4 14. g3 Qh3 15. Be3 Bg4 16. Qd3 Rae8 17. Nd2 Re6
1. Nf3 c5 2. e4 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 Nh5 13. f4 exf4 14. Bxf4 Nc5 15. Qd4 Nxb3+ 16. axb3 Bxb3 17. Bd3 Qb6 18. Qxb6 Nxb6
"""


@pytest.fixture
def corpus_file():
    """Cria um arquivo temporário com partidas PGN."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_GAMES)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def trained_tokenizer(corpus_file):
    """Cria e treina um tokenizador BPE na amostra de PGN."""
    tok = ChessTokenizerBPE(vocab_size=512)
    tok.train(corpus_file)
    return tok


class TestChessTokenizerBPE:
    def test_init(self):
        tok = ChessTokenizerBPE(vocab_size=512)
        assert tok._vocab_size == 512
        assert tok.pad_id is None  # ainda não treinado

    def test_train_and_vocab_size(self, trained_tokenizer):
        tok = trained_tokenizer
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3
        assert tok.vocab_size >= 260  # 256 bytes + 4 especiais

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        tok = trained_tokenizer
        test = "1. e4 e5 2. Nf3 Nc6 3. Bb5+ a6"
        ids = tok.encode(test)
        decoded = tok.decode(ids)
        assert decoded == test, f"Roundtrip falhou: '{decoded}' != '{test}'"

    def test_encode_with_special_tokens(self, trained_tokenizer):
        tok = trained_tokenizer
        test = "e4"
        ids = tok.encode(test, add_special_tokens=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        # Conteúdo entre BOS e EOS deve ser o encode normal
        inner = tok.encode(test)
        assert ids[1:-1] == inner

    def test_decode_skips_special(self, trained_tokenizer):
        tok = trained_tokenizer
        ids = [tok.bos_id, *tok.encode("e4"), tok.eos_id, tok.pad_id]
        decoded = tok.decode(ids, skip_special=True)
        assert decoded == "e4"

    def test_decode_with_special_ids(self, trained_tokenizer):
        tok = trained_tokenizer
        text = "e4"
        inner = tok.encode(text)
        # Decode com BOS/EOS no meio dos IDs (skip_special=True por padrão)
        ids = [tok.bos_id, *inner, tok.eos_id]
        decoded = tok.decode(ids)
        assert decoded == text  # BOS/EOS são pulados

        # skip_special=False não filtra, mas ByteLevel não produz
        # saída visível para tokens especiais
        decoded_raw = tok.decode(ids, skip_special=False)
        assert decoded_raw is not None  # não deve lançar erro

    def test_save_load(self, trained_tokenizer):
        tok = trained_tokenizer
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            tok.save(save_path)
            tok2 = ChessTokenizerBPE.load(save_path)

            assert tok2.vocab_size == tok.vocab_size
            assert tok2.pad_id == tok.pad_id
            assert tok2.unk_id == tok.unk_id
            assert tok2.bos_id == tok.bos_id
            assert tok2.eos_id == tok.eos_id

            # Verifica encode/decode consistente
            test = "1. d4 d5 2. c4 e6"
            assert tok2.encode(test) == tok.encode(test)
            assert tok2.decode(tok2.encode(test)) == test
        finally:
            os.unlink(save_path)

    def test_unknown_chars(self, trained_tokenizer):
        """Caracteres que não existem no corpus podem virar UNK ou bytes."""
        tok = trained_tokenizer
        ids = tok.encode("xyz")  # 'x', 'y', 'z' podem não estar no vocabulário
        decoded = tok.decode(ids)
        # Com ByteLevel, todos os bytes são representáveis, então deve reter
        assert decoded == "xyz"

    def test_empty_string(self, trained_tokenizer):
        tok = trained_tokenizer
        ids = tok.encode("")
        assert ids == []
        assert tok.decode([]) == ""

    def test_full_pgn_game(self, trained_tokenizer):
        """Codifica e decodifica uma partida completa de PGN."""
        tok = trained_tokenizer
        game = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O"
        ids = tok.encode(game)
        decoded = tok.decode(ids)
        assert decoded == game

    def test_castling(self, trained_tokenizer):
        tok = trained_tokenizer
        for castle in ["O-O", "O-O-O"]:
            ids = tok.encode(castle)
            decoded = tok.decode(ids)
            assert decoded == castle

    def test_check_and_mate_symbols(self, trained_tokenizer):
        tok = trained_tokenizer
        for move in ["Bb5+", "Qh7#", "Nf3+"]:
            ids = tok.encode(move)
            decoded = tok.decode(ids)
            assert decoded == move

    def test_promotion(self, trained_tokenizer):
        tok = trained_tokenizer
        move = "e8=Q"
        ids = tok.encode(move)
        decoded = tok.decode(ids)
        assert decoded == move

    def test_result_notations(self, trained_tokenizer):
        tok = trained_tokenizer
        for result in ["1-0", "0-1", "1/2-1/2", "*"]:
            ids = tok.encode(result)
            decoded = tok.decode(ids)
            assert decoded == result

    def test_compression_ratio(self, trained_tokenizer):
        """BPE deve comprimir melhor ou igual ao char-level."""
        from model.tokenizer import ChessTokenizer

        tok_char = ChessTokenizer()
        tok_bpe = trained_tokenizer

        test = "1. e4 e5 2. Nf3 Nc6 3. Bb5+ a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O"
        ids_char = tok_char.encode(test)
        ids_bpe = tok_bpe.encode(test)

        ratio = len(ids_char) / len(ids_bpe)
        assert ratio >= 1.0, (
            f"BPE não comprimiu! char={len(ids_char)}, bpe={len(ids_bpe)}, ratio={ratio:.2f}"
        )

    def test_vocab_size_after_train(self, corpus_file):
        """Treina com vocab_size menor que 260 deve resultar em >= 260 (bytes + especiais)."""
        tok = ChessTokenizerBPE(vocab_size=128)
        tok.train(corpus_file)
        assert tok.vocab_size >= 260

    def test_deterministic_encoding(self, trained_tokenizer):
        """Mesmo texto deve produzir mesmos IDs."""
        tok = trained_tokenizer
        text = "1. e4 e5 2. Nf3"
        ids1 = tok.encode(text)
        ids2 = tok.encode(text)
        assert ids1 == ids2

    def test_repr(self, trained_tokenizer):
        assert "ChessTokenizerBPE" in repr(trained_tokenizer)
        assert str(trained_tokenizer.vocab_size) in repr(trained_tokenizer)
