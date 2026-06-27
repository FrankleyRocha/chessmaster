"""
Fine-tuning do ChessLM nas partidas dos jogadores históricos.

Carrega um checkpoint de pré-treino e continua o treino
com learning rate menor no dataset dos seis jogadores.

Uso:
    python training/finetune.py
    python training/finetune.py --checkpoint checkpoints/pretrain_final.pt
    python training/finetune.py --max-iters 3000 --lr 1e-5
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import ModelConfig, FinetuneConfig
from model.model import ChessLM
from training.train import train


def load_pretrained(checkpoint_path: str, device: str) -> tuple[ChessLM, ModelConfig]:
    """Carrega modelo pré-treinado de um checkpoint."""
    print(f"Carregando checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_model = ModelConfig(**ckpt["cfg_model"])
    model     = ChessLM(cfg_model).to(device)
    
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    print(f"  Checkpoint iter: {ckpt['iter']}")
    print(f"  Val loss pré-treino: {ckpt['val_loss']:.4f}")
    return model, cfg_model


def finetune(checkpoint_path: str, cfg: FinetuneConfig):
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        cfg.device = "cpu"

    model, cfg_model = load_pretrained(checkpoint_path, device)

    print(f"\nIniciando fine-tuning...")
    print(f"  Dataset: {cfg.dataset_name}")
    print(f"  LR: {cfg.learning_rate} (min: {cfg.min_lr})")
    print(f"  Iters: {cfg.max_iters}")

    # Reutiliza o loop de treino padrão com as configs de fine-tuning
    # O modelo já está carregado, então fazemos override manual
    import numpy as np
    import math
    import time
    from training.train import DataLoader, get_lr
    from model.tokenizer import ChessTokenizer

    data_dir = Path(cfg.data_dir)
    name     = cfg.dataset_name

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
    print(f"\nTrain: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    train_loader = DataLoader(train_data, cfg_model.block_size, cfg.batch_size, device)
    val_loader   = DataLoader(val_data,   cfg_model.block_size, cfg.batch_size, device)

    optimizer = model.configure_optimizers(cfg)
    ckpt_dir  = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(cfg.dtype, torch.float32)
    ctx   = torch.amp.autocast(device_type=device, dtype=dtype) if device == "cuda" \
            else torch.amp.autocast(device_type="cpu", dtype=torch.float32)

    best_val_loss = float("inf")
    model.train()
    t0 = time.time()

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg.eval_interval == 0:
            model.eval()
            train_losses, val_losses = [], []
            with torch.no_grad():
                for _ in range(cfg.eval_iters):
                    x, y = train_loader.get_batch()
                    with ctx:
                        _, loss = model(x, y)
                    train_losses.append(loss.item())
                for _ in range(cfg.eval_iters):
                    x, y = val_loader.get_batch()
                    with ctx:
                        _, loss = model(x, y)
                    val_losses.append(loss.item())

            tl = sum(train_losses) / len(train_losses)
            vl = sum(val_losses)   / len(val_losses)
            print(f"iter {it:5d} | train {tl:.4f} | val {vl:.4f} | lr {lr:.2e} | {time.time()-t0:.1f}s")

            if vl < best_val_loss:
                best_val_loss = vl
                ckpt = {
                    "iter":      it,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss":  vl,
                    "cfg_model": cfg_model.__dict__,
                    "cfg_train": cfg.__dict__,
                }
                path = ckpt_dir / f"{cfg.checkpoint_name}_best.pt"
                torch.save(ckpt, path)
                print(f"  ✓ Checkpoint: {path}")
            model.train()

        if it == cfg.max_iters:
            break

        x, y = train_loader.get_batch()
        with ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if it % cfg.log_interval == 0:
            print(f"  step {it:5d} | loss {loss.item():.4f}")

    # Salva final
    path = ckpt_dir / f"{cfg.checkpoint_name}_final.pt"
    torch.save({"iter": cfg.max_iters, "model": model.state_dict(),
                "val_loss": best_val_loss, "cfg_model": cfg_model.__dict__}, path)
    print(f"\nFine-tuning concluído. Checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning do ChessLM")
    parser.add_argument("--checkpoint", default="checkpoints/pretrain_final.pt")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--max-iters",  type=int,   default=5_000)
    parser.add_argument("--lr",         type=float, default=3e-5)
    args = parser.parse_args()

    cfg = FinetuneConfig()
    cfg.device        = args.device
    cfg.max_iters     = args.max_iters
    cfg.learning_rate = args.lr

    finetune(args.checkpoint, cfg)


if __name__ == "__main__":
    main()
