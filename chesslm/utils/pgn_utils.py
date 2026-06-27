"""
Utilitários para parsing e limpeza de arquivos PGN.
"""

import re
from pathlib import Path
from typing import Iterator


# Regex para extrair metadados do PGN
_RE_TAG        = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
_RE_MOVES      = re.compile(r'\{[^}]*\}')      # remove comentários {..}
_RE_EVAL       = re.compile(r'\$\d+')          # remove anotações NAG ($1, $2...)
_RE_RESULT     = re.compile(r'(1-0|0-1|1/2-1/2|\*)\s*$')
_RE_ANNOTATION = re.compile(r'[!?]+')          # remove anotações de lance (!?, ??, etc)
_RE_BLACK_NUM  = re.compile(r'\d+\.\.\.')      # remove numeração das pretas (1..., 2...)


def parse_pgn_file(path: str) -> Iterator[dict]:
    """
    Lê um arquivo .pgn e yield cada partida como dicionário:
        {
          'tags':   {'White': ..., 'Black': ..., 'WhiteElo': ..., 'Result': ...},
          'moves':  '1. e4 e5 2. Nf3 ...',
        }
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Divide o arquivo em blocos por partida
    games = re.split(r'\n\n(?=\[)', content.strip())

    for block in games:
        block = block.strip()
        if not block:
            continue

        tags = {}
        for match in _RE_TAG.finditer(block):
            tags[match.group(1)] = match.group(2)

        # Extrai a seção de movimentos (depois dos tags)
        move_section = re.sub(r'\[.*?\]\s*', '', block, flags=re.DOTALL).strip()

        if not move_section:
            continue

        moves = clean_moves(move_section)
        if not moves:
            continue

        yield {"tags": tags, "moves": moves}


def clean_moves(raw: str) -> str:
    """
    Limpa a string de movimentos:
    - Remove comentários {...}
    - Remove anotações NAG ($1, $2...)
    - Remove anotações de lance (!, ?, !!, ??, !?, ?!)
    - Remove numeração das pretas (1..., 2...)
    - Remove resultado final (1-0, 0-1, 1/2-1/2)
    - Normaliza espaços (remove espaço após número das brancas: "1. " -> "1.")
    """
    s = _RE_MOVES.sub("", raw)           # remove comentários
    s = _RE_EVAL.sub("", s)              # remove NAGs
    s = _RE_ANNOTATION.sub("", s)        # remove anotações de lance (!?, etc)
    s = _RE_BLACK_NUM.sub("", s)         # remove numeração das pretas
    s = _RE_RESULT.sub("", s)            # remove resultado
    s = re.sub(r'(\d+)\. ', r'\1.', s)   # remove espaço após número: "1. " -> "1."
    s = re.sub(r'\s+', ' ', s)           # normaliza espaços
    return s.strip()


def filter_game(tags: dict, min_elo: int = 2500, only_decisive: bool = True) -> bool:
    """
    Retorna True se a partida passa nos filtros definidos.

    Args:
        tags:           dicionário de metadados do PGN
        min_elo:        Elo mínimo de AMBOS os jogadores
        only_decisive:  se True, descarta empates
    """
    # Filtra por Elo
    try:
        white_elo = int(tags.get("WhiteElo", 0))
        black_elo = int(tags.get("BlackElo", 0))
    except (ValueError, TypeError):
        return False

    if white_elo < min_elo or black_elo < min_elo:
        return False

    # Filtra por resultado
    result = tags.get("Result", "")
    if only_decisive and result not in ("1-0", "0-1"):
        return False

    return True


def extract_winner_moves(game: dict) -> str | None:
    """
    Extrai os movimentos do vencedor como sequência de tokens.
    Para partidas 1-0 retorna movimentos das brancas,
    para 0-1 retorna movimentos das pretas.

    Retorna None se não for possível determinar.
    """
    result = game["tags"].get("Result", "")
    moves_str = game["moves"]

    # Parseia a lista de movimentos
    tokens = moves_str.split()
    white_moves = []
    black_moves = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Número do movimento (ex: "1.", "2.")
        if re.match(r'^\d+\.$', token):
            if i + 1 < len(tokens):
                white_moves.append(tokens[i + 1])
            if i + 2 < len(tokens) and not re.match(r'^\d+', tokens[i + 2]):
                black_moves.append(tokens[i + 2])
            i += 3
        else:
            i += 1

    if result == "1-0":
        return " ".join(white_moves)
    elif result == "0-1":
        return " ".join(black_moves)
    return None


def pgn_to_text(game: dict) -> str:
    """Converte uma partida para o formato de treino (string PGN limpa)."""
    return game["moves"]


if __name__ == "__main__":
    # Teste rápido com PGN sintético
    sample = """[Event "Test"]
[White "Kasparov"]
[Black "Karpov"]
[WhiteElo "2800"]
[BlackElo "2750"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 { Abertura Ruy Lopez } a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
        f.write(sample)
        tmp = f.name

    for game in parse_pgn_file(tmp):
        print("Tags:", game["tags"])
        print("Moves:", game["moves"])
        print("Passa filtro:", filter_game(game["tags"]))

    os.unlink(tmp)
