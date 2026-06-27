"""
Geração de movimentos com o ChessLM.

Dado um prompt em notação PGN, o modelo gera os próximos movimentos.

Uso:
    python inference/generate.py --prompt "1. e4 e5 2. Nf3"
    python inference/generate.py --prompt "1. d4 d5" --moves 10 --temperature 0.8
    python inference/generate.py --checkpoint checkpoints/finetune_best.pt --prompt "1. e4"
    python inference/generate.py --interactive
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import ModelConfig
from model.model import ChessLM
from model.tokenizer import ChessTokenizer


def load_model(checkpoint_path: str, device: str) -> tuple[ChessLM, ChessTokenizer]:
    """Carrega modelo e tokenizador de um checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg   = ModelConfig(**ckpt["cfg_model"])
    model = ChessLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Tenta carregar tokenizador do diretório do checkpoint
    ckpt_dir = Path(checkpoint_path).parent
    tok_path = ckpt_dir.parent / "data" / "tokenizer.json"
    if not tok_path.exists():
        tok_path = ckpt_dir / "tokenizer.json"

    if tok_path.exists():
        tok = ChessTokenizer.load(str(tok_path))
    else:
        tok = ChessTokenizer()
        print("Aviso: tokenizer.json não encontrado, usando padrão")

    return model, tok


def generate_moves(
    model: ChessLM,
    tok: ChessTokenizer,
    prompt: str,
    num_moves: int = 5,
    temperature: float = 1.0,
    top_k: int = 10,
    device: str = "cpu",
) -> str:
    """
    Gera `num_moves` movimentos a partir do prompt.

    Estratégia: gera caractere por caractere até detectar
    `num_moves` movimentos completos (separados por espaços
    após os números de movimento).
    """
    import re

    # Conta quantos movimentos já existem no prompt
    existing = len(re.findall(r'\d+\.', prompt))
    target   = existing + num_moves

    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generated = prompt
    max_chars  = num_moves * 15  # estimativa conservadora de chars por movimento

    with torch.no_grad():
        for _ in range(max_chars):
            idx_gen = model.generate(idx, max_new_tokens=1,
                                     temperature=temperature, top_k=top_k)
            new_id  = idx_gen[0, -1].item()
            char    = tok.decode([new_id])
            generated += char
            idx = idx_gen

            # Verifica se atingimos o número de movimentos alvo
            current = len(re.findall(r'\d+\.', generated))
            if current > target:
                # Remove o último número de movimento incompleto
                generated = generated[:generated.rfind(str(current) + ".")].rstrip()
                break

    return generated


def interactive_mode(model: ChessLM, tok: ChessTokenizer, device: str):
    """Modo interativo: usuário digita a partida, modelo responde."""
    print("\n" + "═" * 60)
    print("  ChessLM — Modo Interativo")
    print("  Digite a partida em PGN ou 'sair' para encerrar")
    print("═" * 60)

    while True:
        try:
            prompt = input("\nPartida > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if prompt.lower() in ("sair", "exit", "quit", "q"):
            break

        if not prompt:
            prompt = "1."

        print("Gerando...", end=" ", flush=True)
        result = generate_moves(model, tok, prompt, num_moves=3, device=device)
        print("\n" + result)


def main():
    parser = argparse.ArgumentParser(description="Geração de movimentos com ChessLM")
    parser.add_argument("--checkpoint",   default="checkpoints/finetune_best.pt")
    parser.add_argument("--prompt",       default="1. e4",
                        help="Partida inicial em PGN")
    parser.add_argument("--moves",        type=int,   default=5,
                        help="Número de movimentos a gerar")
    parser.add_argument("--temperature",  type=float, default=0.8,
                        help="Temperatura (0.5=conservador, 1.5=criativo)")
    parser.add_argument("--top-k",        type=int,   default=10,
                        help="Top-k sampling")
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--interactive",  action="store_true",
                        help="Modo interativo")
    args = parser.parse_args()

    print(f"Carregando {args.checkpoint}...")
    model, tok = load_model(args.checkpoint, args.device)
    print(f"Modelo carregado — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n")

    if args.interactive:
        interactive_mode(model, tok, args.device)
        return

    print(f"Prompt:    {args.prompt}")
    print(f"Gerando {args.moves} movimentos (temp={args.temperature}, top_k={args.top_k})...\n")

    result = generate_moves(
        model, tok,
        prompt      = args.prompt,
        num_moves   = args.moves,
        temperature = args.temperature,
        top_k       = args.top_k,
        device      = args.device,
    )

    print("Resultado:")
    print(result)


if __name__ == "__main__":
    main()
