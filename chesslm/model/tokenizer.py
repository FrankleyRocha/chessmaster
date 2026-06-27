"""
Factory de tokenizadores — retorna o tokenizador adequado baseado no tipo.

Uso:
    tok = create_tokenizer("bpe")
    tok = create_tokenizer("char")
    tok = load_tokenizer("data/tokenizer_bpe.json", "bpe")
"""

from model.tokenizer_char import ChessTokenizerChar
from model.tokenizer_bpe import ChessTokenizerBPE


def create_tokenizer(tokenizer_type: str = "bpe", **kwargs):
    """Cria e retorna um tokenizador do tipo especificado.

    Args:
        tokenizer_type: "bpe" ou "char"
        **kwargs: passados para o construtor do tokenizador

    Returns:
        ChessTokenizerBPE ou ChessTokenizerChar
    """
    if tokenizer_type == "bpe":
        return ChessTokenizerBPE(**kwargs)
    elif tokenizer_type == "char":
        return ChessTokenizerChar(**kwargs)
    raise ValueError(f"Tipo de tokenizador desconhecido: {tokenizer_type}")


def load_tokenizer(path: str, tokenizer_type: str = "bpe"):
    """Carrega um tokenizador salvo em disco.

    Args:
        path: Caminho do arquivo JSON do tokenizador
        tokenizer_type: "bpe" ou "char"

    Returns:
        ChessTokenizerBPE ou ChessTokenizerChar carregado
    """
    if tokenizer_type == "bpe":
        return ChessTokenizerBPE.load(path)
    elif tokenizer_type == "char":
        return ChessTokenizerChar.load(path)
    raise ValueError(f"Tipo de tokenizador desconhecido: {tokenizer_type}")
