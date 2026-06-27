"""
Download e filtragem do Lichess Database.

Baixa dumps mensais em .pgn.zst, filtra partidas com Elo >= 2500
e apenas vitórias (sem empates), e salva em texto plano para treino.

Uso:
    python data/download_lichess.py --month 2024-01 --output data/pretrain.txt
    python data/download_lichess.py --month 2024-01 --max-games 200000
"""

import argparse
import sys
import zstandard as zstd
import io
import re
import requests
from pathlib import Path
from tqdm import tqdm

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pgn_utils import parse_pgn_file, filter_game, clean_moves

BASE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"

# Tags necessárias do PGN
_RE_TAG    = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
_RE_RESULT = re.compile(r'^(1-0|0-1|1/2-1/2|\*)')


def stream_lichess(month: str, min_elo: int, only_decisive: bool, max_games: int, output: str):
    """
    Faz streaming do dump do Lichess, filtra e salva em output.
    Não precisa baixar o arquivo completo (~GB) antes de processar.
    """
    url = BASE_URL.format(month=month)
    print(f"Conectando a {url}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    games_written = 0
    games_seen    = 0
    buffer        = []

    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        pbar  = tqdm(total=total, unit="B", unit_scale=True, desc="Download")

        with open(output_path, "w", encoding="utf-8") as out:
            with dctx.stream_reader(r.raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

                game_lines = []
                in_moves   = False
                tags       = {}

                for line in text_stream:
                    pbar.update(len(line.encode("utf-8")))
                    line = line.rstrip()

                    if line.startswith("["):
                        m = _RE_TAG.match(line)
                        if m:
                            tags[m.group(1)] = m.group(2)
                        game_lines.append(line)

                    elif line == "":
                        if in_moves and game_lines:
                            # Fim da partida — processa
                            games_seen += 1
                            if filter_game(tags, min_elo=min_elo, only_decisive=only_decisive):
                                moves_raw = " ".join(buffer)
                                moves = clean_moves(moves_raw)
                                if moves:
                                    out.write(moves + "\n")
                                    games_written += 1

                                    if games_written % 10_000 == 0:
                                        print(f"\n  {games_written:,} partidas salvas "
                                              f"({games_seen:,} processadas)")

                                    if max_games and games_written >= max_games:
                                        pbar.close()
                                        print(f"\nLimite de {max_games:,} partidas atingido.")
                                        print(f"Total: {games_written:,} salvas de {games_seen:,} processadas")
                                        return

                            # Reset
                            game_lines = []
                            buffer     = []
                            tags       = {}
                            in_moves   = False

                    else:
                        # Linha de movimentos
                        in_moves = True
                        buffer.append(line)
                        game_lines.append(line)

        pbar.close()

    print(f"\nConcluído: {games_written:,} partidas salvas de {games_seen:,} processadas")
    print(f"Arquivo: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download e filtragem do Lichess Database")
    parser.add_argument("--month",         default="2024-01",
                        help="Mês do dump (formato YYYY-MM). Default: 2024-01")
    parser.add_argument("--output",        default="data/pretrain.txt",
                        help="Arquivo de saída. Default: data/pretrain.txt")
    parser.add_argument("--min-elo",       type=int, default=2500,
                        help="Elo mínimo para ambos os jogadores. Default: 2500")
    parser.add_argument("--max-games",     type=int, default=0,
                        help="Limite de partidas (0 = sem limite). Default: 0")
    parser.add_argument("--keep-draws",    action="store_true",
                        help="Inclui empates (padrão: apenas vitórias)")
    args = parser.parse_args()

    stream_lichess(
        month         = args.month,
        min_elo       = args.min_elo,
        only_decisive = not args.keep_draws,
        max_games     = args.max_games,
        output        = args.output,
    )


if __name__ == "__main__":
    main()
