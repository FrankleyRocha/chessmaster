"""
Download das partidas dos jogadores históricos do PGN Mentor.

Baixa arquivos .pgn para Fischer, Kasparov, Tal, Capablanca,
Praggnanandhaa e Magnus Carlsen e concatena num único arquivo de treino.

Uso:
    python data/download_players.py --output data/players.txt
    python data/download_players.py --players kasparov tal --output data/kasparov_tal.txt
"""

import argparse
import sys
import requests
import tempfile
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pgn_utils import parse_pgn_file, clean_moves

# URLs dos arquivos PGN no PGN Mentor
PLAYER_URLS = {
    "fischer": {
        "name": "Bobby Fischer",
        "url": "https://www.pgnmentor.com/players/Fischer.zip",
        "zip": True,
        "file": "Fischer.pgn",
    },
    "kasparov": {
        "name": "Garry Kasparov",
        "url": "https://www.pgnmentor.com/players/Kasparov.zip",
        "zip": True,
        "file": "Kasparov.pgn",
    },
    "tal": {
        "name": "Mikhail Tal",
        "url": "https://www.pgnmentor.com/players/Tal.zip",
        "zip": True,
        "file": "Tal.pgn",
    },
    "capablanca": {
        "name": "José Capablanca",
        "url": "https://www.pgnmentor.com/players/Capablanca.zip",
        "zip": True,
        "file": "Capablanca.pgn",
    },
    "pragg": {
        "name": "Praggnanandhaa",
        "url": "https://www.pgnmentor.com/players/Praggnanandhaa.zip",
        "zip": True,
        "file": "Praggnanandhaa.pgn",
    },
    "magnus": {
        "name": "Magnus Carlsen",
        "url": "https://www.pgnmentor.com/players/Carlsen.zip",
        "zip": True,
        "file": "Carlsen.pgn",
    },
}


def download_file(url: str, dest: str):
    """Download com barra de progresso."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.pgnmentor.com/",
    }
    r = requests.get(url, stream=True, headers=headers)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: str, filename: str, dest: str):
    """Extrai arquivo específico de um ZIP."""
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        # Tenta encontrar o arquivo (pode estar numa subpasta)
        candidates = [n for n in z.namelist() if n.endswith(filename)]
        if not candidates:
            raise FileNotFoundError(f"{filename} não encontrado em {zip_path}")
        with z.open(candidates[0]) as src, open(dest, "wb") as dst:
            dst.write(src.read())


def process_player(key: str, info: dict, output_file, stats: dict):
    """Baixa, extrai e processa as partidas de um jogador."""
    print(f"\n{'─'*50}")
    print(f"  {info['name']}")
    print(f"{'─'*50}")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, f"{key}.zip")
        pgn_path = os.path.join(tmpdir, f"{key}.pgn")

        # Download
        print(f"  Baixando de {info['url']}")
        try:
            download_file(info["url"], zip_path)
        except Exception as e:
            print(f"  ✗ Erro no download: {e}")
            return

        # Extrai PGN do ZIP
        if info.get("zip"):
            try:
                extract_zip(zip_path, info["file"], pgn_path)
            except Exception as e:
                print(f"  ✗ Erro na extração: {e}")
                return
        else:
            pgn_path = zip_path

        # Processa partidas
        count = 0
        for game in parse_pgn_file(pgn_path):
            moves = game["moves"]
            if moves and len(moves) > 10:  # ignora partidas muito curtas
                output_file.write(moves + "\n")
                count += 1

        stats[info["name"]] = count
        print(f"  ✓ {count:,} partidas processadas")


def main():
    parser = argparse.ArgumentParser(description="Download de partidas dos jogadores históricos")
    parser.add_argument("--output",  default="data/players.txt",
                        help="Arquivo de saída. Default: data/players.txt")
    parser.add_argument("--players", nargs="+",
                        choices=list(PLAYER_URLS.keys()),
                        default=list(PLAYER_URLS.keys()),
                        help="Jogadores a baixar. Default: todos")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {}
    print(f"Salvando em: {output_path}")

    with open(output_path, "w", encoding="utf-8") as out:
        for key in args.players:
            info = PLAYER_URLS[key]
            process_player(key, info, out, stats)

    # Resumo
    print(f"\n{'═'*50}")
    print("  Resumo")
    print(f"{'═'*50}")
    total = 0
    for name, count in stats.items():
        print(f"  {name:<25} {count:>6,} partidas")
        total += count
    print(f"{'─'*50}")
    print(f"  {'Total':<25} {total:>6,} partidas")
    print(f"\nArquivo: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
