"""
Loop de pré-treino do ChessLM.

Uso:
    python training/train.py
    python training/train.py --device cpu
    python training/train.py --max-iters 10000 --batch-size 32
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import ModelConfig, TrainConfig
from model.model import ChessLM
from model.tokenizer import load_tokenizer


# ─────────────────────────────────────────────
#  Utilitários
# ─────────────────────────────────────────────

def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay com linear warmup."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # Cosine
    progress = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


class DataLoader:
    """Carrega batches aleatórios de um array numpy."""

    def __init__(self, data: np.ndarray, block_size: int, batch_size: int, device: str):
        self.data       = torch.from_numpy(data.astype(np.int64))
        self.block_size = block_size
        self.batch_size = batch_size
        self.device     = device

    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x  = torch.stack([self.data[i     : i + self.block_size    ] for i in ix])
        y  = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


# ─────────────────────────────────────────────
#  Treino
# ─────────────────────────────────────────────

def train(cfg_model: ModelConfig, cfg_train: TrainConfig):
    # Setup device
    device = cfg_train.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA não disponível, usando CPU")
        device = "cpu"
        cfg_train.device = "cpu"

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(cfg_train.dtype, torch.float32)
    ctx   = torch.amp.autocast(device_type=device, dtype=dtype) if device == "cuda" \
            else torch.amp.autocast(device_type="cpu", dtype=torch.float32)

    # Dados
    data_dir = Path(cfg_train.data_dir)
    name     = cfg_train.dataset_name

    train_path = data_dir / f"{name}_train.npy"
    val_path   = data_dir / f"{name}_val.npy"

    if not train_path.exists() or not val_path.exists():
        print(f"\n✗ Erro: Arquivos de dataset não encontrados!")
        print(f"  Esperado: {train_path} e {val_path}")
        print(f"\n  Rode primeiro:")
        print(f"    python data/prepare_dataset.py --input data/{name}.txt --name {name}")
        sys.exit(1)

    train_data = np.load(train_path)
    val_data   = np.load(val_path)

    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    train_loader = DataLoader(train_data, cfg_model.block_size, cfg_train.batch_size, device)
    val_loader   = DataLoader(val_data,   cfg_model.block_size, cfg_train.batch_size, device)

    # Tokenizador (para vocab_size)
    tok_filename = f"tokenizer_{cfg_model.tokenizer_type}.json"
    tok_path = data_dir / tok_filename
    if not tok_path.exists():
        print(f"\n✗ Erro: Tokenizador não encontrado: {tok_path}")
        print(f"  Rode primeiro: python data/prepare_dataset.py --tokenizer-type {cfg_model.tokenizer_type}")
        sys.exit(1)
    tok = load_tokenizer(str(tok_path), cfg_model.tokenizer_type)
    cfg_model.vocab_size = tok.vocab_size

    # Modelo
    model = ChessLM(cfg_model).to(device)
    if cfg_train.compile and device == "cuda":
        print("Compilando modelo com torch.compile...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(cfg_train)
    scaler    = torch.cuda.GradScaler() if device == "cuda" and dtype == torch.float16 else None

    # Checkpoint dir
    ckpt_dir = Path(cfg_train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Loop de treino ──
    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(cfg_train.max_iters + 1):

        # Atualiza lr
        lr = get_lr(it, cfg_train)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Avaliação ──
        if it % cfg_train.eval_interval == 0:
            model.eval()
            losses = {"train": [], "val": []}
            with torch.no_grad():
                for split, loader in [("train", train_loader), ("val", val_loader)]:
                    for _ in range(cfg_train.eval_iters):
                        x, y = loader.get_batch()
                        with ctx:
                            _, loss = model(x, y)
                        losses[split].append(loss.item())

            train_loss = sum(losses["train"]) / len(losses["train"])
            val_loss   = sum(losses["val"])   / len(losses["val"])
            elapsed    = time.time() - t0

            print(f"iter {it:6d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| lr {lr:.2e} | {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "iter":       it,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_loss":   val_loss,
                    "cfg_model":  cfg_model.__dict__,
                    "cfg_train":  cfg_train.__dict__,
                }
                path = ckpt_dir / f"{cfg_train.checkpoint_name}_best.pt"
                torch.save(ckpt, path)
                print(f"  ✓ Checkpoint salvo: {path}")

            model.train()

        if it == cfg_train.max_iters:
            break

        # ── Passo de treino ──
        x, y = train_loader.get_batch()

        with ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            optimizer.step()

        if it % cfg_train.log_interval == 0:
            print(f"  step {it:6d} | loss {loss.item():.4f} | lr {lr:.2e}")

        # Salva checkpoint periódico
        if it > 0 and it % cfg_train.save_interval == 0:
            ckpt = {
                "iter":      it,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  None,
                "cfg_model": cfg_model.__dict__,
                "cfg_train": cfg_train.__dict__,
            }
            path = ckpt_dir / f"{cfg_train.checkpoint_name}_iter{it}.pt"
            torch.save(ckpt, path)

    # Salva checkpoint final
    ckpt = {
        "iter":      cfg_train.max_iters,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss":  best_val_loss,
        "cfg_model": cfg_model.__dict__,
        "cfg_train": cfg_train.__dict__,
    }
    path = ckpt_dir / f"{cfg_train.checkpoint_name}_final.pt"
    torch.save(ckpt, path)
    print(f"\nTreino concluído. Checkpoint final: {path}")


def main():
    parser = argparse.ArgumentParser(description="Pré-treino do ChessLM")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--max-iters",     type=int,   default=50_000)
    parser.add_argument("--no-compile",    action="store_true")
    parser.add_argument("--tokenizer-type", default="word", choices=["bpe", "char", "word"],
                        help="Tipo de tokenizador. Default: word")
    args = parser.parse_args()

    cfg_model = ModelConfig(tokenizer_type=args.tokenizer_type)
    cfg_train = TrainConfig()
    cfg_train.device     = args.device
    cfg_train.batch_size = args.batch_size
    cfg_train.max_iters  = args.max_iters
    cfg_train.compile    = not args.no_compile

    train(cfg_model, cfg_train)


if __name__ == "__main__":
    main()
