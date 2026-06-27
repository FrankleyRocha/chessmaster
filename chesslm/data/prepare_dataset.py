"""
Prepara os datasets para treino:
- Tokeniza por caractere
- Divide em train/val
- Serializa como arrays numpy binários (.npy)

Uso:
    python data/prepare_dataset.py --input data/pretrain.txt --name pretrain
    python data/prepare_dataset.py --input data/players.txt  --name finetune
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import create_tokenizer


def tokenizer_path_for(tokenizer_type: str) -> str:
    return f"tokenizer_{tokenizer_type}.json"


def prepare(input_path: str, name: str, val_split: float = 0.05,
            tokenizer_type: str = "bpe", bpe_vocab_size: int = 512):
    input_path = Path(input_path)
    out_dir    = input_path.parent

    print(f"Lendo {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"  Caracteres total: {len(text):,}")

    kwargs = {}
    if tokenizer_type == "bpe":
        kwargs["vocab_size"] = bpe_vocab_size

    print(f"Criando tokenizador ({tokenizer_type})...")
    tok = create_tokenizer(tokenizer_type, **kwargs)
    tok.train(str(input_path))

    tok_filename = tokenizer_path_for(tokenizer_type)
    tok_path = out_dir / tok_filename
    tok.save(str(tok_path))
    print(f"  Tokenizador: vocab_size={tok.vocab_size}")

    # Tokeniza em batches para não estourar memória
    print("Tokenizando...")
    lines = text.split("\n")
    all_ids = []
    for line in tqdm(lines, unit="linhas"):
        if line.strip():
            ids = tok.encode(line + "\n", add_special_tokens=False)
            all_ids.extend(ids)

    data = np.array(all_ids, dtype=np.uint16)
    print(f"  Tokens total: {len(data):,}")

    # Split train/val
    n_val   = int(len(data) * val_split)
    n_train = len(data) - n_val

    train_data = data[:n_train]
    val_data   = data[n_train:]

    train_path = out_dir / f"{name}_train.npy"
    val_path   = out_dir / f"{name}_val.npy"

    np.save(str(train_path), train_data)
    np.save(str(val_path),   val_data)

    print(f"\nDataset salvo:")
    print(f"  train: {len(train_data):,} tokens → {train_path}")
    print(f"  val:   {len(val_data):,} tokens  → {val_path}")
    print(f"  ratio: {len(val_data)/len(data)*100:.1f}% val")


def main():
    parser = argparse.ArgumentParser(description="Prepara dataset para treino")
    parser.add_argument("--input",          required=True,
                        help="Arquivo .txt com partidas (uma por linha)")
    parser.add_argument("--name",           default="dataset",
                        help="Prefixo dos arquivos de saída. Default: dataset")
    parser.add_argument("--val-split",      type=float, default=0.05,
                        help="Fração para validação. Default: 0.05")
    parser.add_argument("--tokenizer-type", default="bpe", choices=["bpe", "char"],
                        help="Tipo de tokenizador. Default: bpe")
    parser.add_argument("--bpe-vocab-size", type=int, default=512,
                        help="Tamanho do vocabulário BPE (só usado se --tokenizer-type=bpe). Default: 512")
    args = parser.parse_args()

    prepare(args.input, args.name, args.val_split, args.tokenizer_type, args.bpe_vocab_size)


if __name__ == "__main__":
    main()
