# Solução: Exercício 4 - Melhorias e Extensões

Soluções completas para o [[exercicios/exercicio-04-melhorias|Exercício 4]].

## Tarefa 1: Adicionar Novo Jogador

```python
#!/usr/bin/env python
"""Adiciona novo jogador ao dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "chesslm"))

from data.download_players import PLAYER_URLS, process_player

def add_new_player():
    """Adiciona Karpov ao dataset."""
    
    # Novo jogador
    karpov = {
        "karpov": {
            "name": "Anatoly Karpov",
            "url": "https://www.pgnmentor.com/players/Karpov.zip",
            "zip": True,
            "file": "Karpov.pgn",
        }
    }
    
    # Adiciona ao dict existente
    PLAYER_URLS.update(karpov)
    
    print("Jogadores disponíveis:")
    for key in PLAYER_URLS:
        print(f"  - {key}: {PLAYER_URLS[key]['name']}")
    
    print("\nPara baixar:")
    print("  python data/download_players.py --players karpov --output data/karpov.txt")

def merge_datasets():
    """Merge de múltiplos jogadores em um dataset."""
    
    import subprocess
    
    players = ["fischer", "karpov", "kasparov"]
    files = [f"data/{p}.txt" for p in players]
    
    # Baixa cada jogador
    for p in players:
        print(f"Baixando {p}...")
        subprocess.run([
            "python", "data/download_players.py",
            "--players", p,
            "--output", f"data/{p}.txt"
        ])
    
    # Merge
    output_file = "data/custom_players.txt"
    
    with open(output_file, "w") as out:
        for f in files:
            if Path(f).exists():
                with open(f) as infile:
                    out.write(infile.read())
    
    print(f"\nDataset criado: {output_file}")

if __name__ == "__main__":
    add_new_player()
```

---

## Tarefa 2: Experimentar Hiperparâmetros

```python
#!/usr/bin/env python
"""Busca por melhores hiperparâmetros."""

import itertools
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "chesslm"))

from model.config import ModelConfig
from model.model import ChessLM

def count_parameters(model) -> float:
    """Conta parâmetros em milhões."""
    return sum(p.numel() for p in model.parameters()) / 1e6

def simulate_training_loss(params: dict, n_iters: int = 1000) -> float:
    """
    Simula loss final baseado nos hiperparâmetros.
    
    Em produção, treinar de verdade!
    """
    
    # Heurística simplificada
    base_loss = 1.5
    
    # Mais capacidade = melhor loss
    capacity_bonus = -0.3 * (params["n_embd"] / 256) * (params["n_layer"] / 6)
    
    # Dropout muito alto = pior
    dropout_penalty = 0.2 * max(0, params["dropout"] - 0.15)
    
    # Block size muito grande sem necessidade = overfitting
    bs_penalty = 0.1 * max(0, params["block_size"] / 512 - 1.5)
    
    final_loss = base_loss + capacity_bonus + dropout_penalty + bs_penalty
    final_loss += np.random.randn() * 0.05  # Ruído
    
    return max(0.5, final_loss)

def hyperparameter_search():
    """Busca sistemática."""
    
    print("Busca de Hiperparâmetros")
    print("="*60)
    
    # Grade
    param_grid = {
        "n_embd": [128, 256, 384],
        "n_layer": [4, 6, 8],
        "n_head": [4, 8],
        "dropout": [0.0, 0.1, 0.2],
        "block_size": [256, 512],
    }
    
    # Gera combinações
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total de combinações: {len(combinations)}")
    
    results = []
    
    for i, params in enumerate(combinations):
        # Validação
        if params["n_embd"] % params["n_head"] != 0:
            continue
        
        # Conta parâmetros
        cfg = ModelConfig(**params)
        model = ChessLM(cfg)
        n_params = count_parameters(model)
        
        # Simula loss
        loss = simulate_training_loss(params)
        
        results.append({
            **params,
            "params_M": round(n_params, 2),
            "val_loss": round(loss, 4),
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processadas {i+1}/{len(combinations)} combinações")
    
    # DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("val_loss")
    
    print("\n" + "="*60)
    print("Top 10 Configurações:")
    print("="*60)
    print(df.head(10).to_string(index=False))
    
    # Salva
    df.to_csv("hyperparameter_search.csv", index=False)
    print("\nResultados salvos: hyperparameter_search.csv")
    
    return df

def analyze_results(df: pd.DataFrame):
    """Analisa importância de cada hiperparâmetro."""
    
    print("\n" + "="*60)
    print("Análise de Importância")
    print("="*60)
    
    for param in ["n_embd", "n_layer", "n_head", "dropout", "block_size"]:
        grouped = df.groupby(param)["val_loss"].mean()
        print(f"\n{param}:")
        print(grouped.to_string())

if __name__ == "__main__":
    np.random.seed(42)
    df = hyperparameter_search()
    analyze_results(df)
```

---

## Tarefa 3: Avaliação com Stockfish

```python
#!/usr/bin/env python
"""Avalia qualidade dos movimentos com Stockfish."""

import chess
import chess.engine
import chess.pgn
import io
from typing import Optional
import numpy as np

class GameEvaluator:
    """Avaliador de partidas usando Stockfish."""
    
    def __init__(self, engine_path: str = "stockfish", depth: int = 15):
        """
        Args:
            engine_path: Caminho para o executável do Stockfish
            depth: Profundidade de análise
        """
        self.engine_path = engine_path
        self.depth = depth
        self.engine = None
    
    def start(self):
        """Inicia o engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            return True
        except Exception as e:
            print(f"Erro ao iniciar Stockfish: {e}")
            return False
    
    def stop(self):
        """Para o engine."""
        if self.engine:
            self.engine.quit()
    
    def evaluate_game(self, pgn: str) -> dict:
        """
        Avalia uma partida.
        
        Args:
            pgn: String PGN da partida
        
        Returns:
            Métricas de qualidade
        """
        if not self.engine:
            if not self.start():
                return {"error": "Engine não disponível"}
        
        # Parse
        try:
            game = chess.pgn.read_game(io.StringIO(pgn))
            board = game.board()
        except Exception as e:
            return {"error": str(e), "valid": False}
        
        cp_losses = []
        blunders = 0
        inaccuracies = 0
        mistakes = 0
        
        prev_score = 0
        move_count = 0
        
        for move in game.mainline_moves():
            # Verifica legalidade
            if move not in board.legal_moves:
                return {
                    "valid": False,
                    "error": f"Movimento ilegal: {move}",
                    "moves": move_count,
                }
            
            # Avalia antes
            info_before = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score_before = info_before["score"].relative.score(mate_score=10000)
            
            # Faz movimento
            board.push(move)
            move_count += 1
            
            # Avalia depois
            info_after = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score_after = -info_after["score"].relative.score(mate_score=10000)  # Negativo
            
            # Calcula perda
            if score_before is not None and score_after is not None:
                cp_loss = score_before - score_after
                cp_losses.append(abs(cp_loss))
                
                # Classifica
                if abs(cp_loss) > 100:
                    blunders += 1
                elif abs(cp_loss) > 50:
                    mistakes += 1
                elif abs(cp_loss) > 20:
                    inaccuracies += 1
            
            prev_score = score_after
        
        # Métricas
        avg_cp_loss = np.mean(cp_losses) if cp_losses else 0
        
        # Accuracy (heurística)
        accuracy = max(0, 100 - avg_cp_loss * 2)
        
        return {
            "valid": True,
            "moves": move_count,
            "avg_cp_loss": round(avg_cp_loss, 2),
            "blunders": blunders,
            "mistakes": mistakes,
            "inaccuracies": inaccuracies,
            "accuracy": round(accuracy, 1),
        }
    
    def evaluate_multiple(self, pgns: list[str]) -> dict:
        """Avalia múltiplas partidas."""
        
        results = []
        
        for pgn in pgns:
            result = self.evaluate_game(pgn)
            results.append(result)
        
        # Filtra válidas
        valid_results = [r for r in results if r.get("valid", False)]
        
        if not valid_results:
            return {"error": "Nenhuma partida válida"}
        
        return {
            "total_games": len(pgns),
            "valid_games": len(valid_results),
            "avg_moves": np.mean([r["moves"] for r in valid_results]),
            "avg_cp_loss": np.mean([r["avg_cp_loss"] for r in valid_results]),
            "total_blunders": sum(r["blunders"] for r in valid_results),
            "avg_accuracy": np.mean([r["accuracy"] for r in valid_results]),
        }

def test_evaluator():
    """Testa o avaliador."""
    
    print("Testando GameEvaluator")
    print("="*50)
    
    evaluator = GameEvaluator(depth=12)
    
    # Partida de teste
    test_pgns = [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O",
    ]
    
    for i, pgn in enumerate(test_pgns, 1):
        print(f"\nPartida {i}:")
        print(f"  PGN: {pgn}")
        
        result = evaluator.evaluate_game(pgn)
        
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    evaluator.stop()

def evaluate_model_output(model, tok, n_games: int = 10):
    """Avalia qualidade do modelo."""
    
    evaluator = GameEvaluator()
    
    prompts = ["1. e4", "1. d4", "1. Nf3", "1. c4", "1. g3"]
    
    generated_pgns = []
    
    for prompt in prompts[:n_games]:
        # Gerar jogo (precisa ter modelo carregado)
        # pgn = generate_moves(model, tok, prompt, num_moves=20)
        # generated_pgns.append(pgn)
        pass
    
    results = evaluator.evaluate_multiple(generated_pgns)
    
    print("\nAvaliação do Modelo:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    evaluator.stop()

if __name__ == "__main__":
    test_evaluator()
```

---

## Tarefa 4: Implementar RoPE

```python
#!/usr/bin/env python
"""Implementação de Rotary Position Embeddings (RoPE)."""

import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Benefícios:
    - Melhor generalização para sequências longas
    - Codificação de posição relativa
    - Usado em LLaMA, Mistral, etc.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pré-computa frequências inversas
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache de cos/sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Constrói cache de cos/sin."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Frequências
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        
        # Duplica para (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Aplica RoPE ao input.
        
        Args:
            x: (batch, seq_len, n_heads, head_dim)
            offset: Offset de posição (para geração incremental)
        
        Returns:
            Tensor com rotação aplicada
        """
        seq_len = x.shape[1]
        
        # Expande cache se necessário
        if offset + seq_len > self.cos_cached.shape[0]:
            self._build_cache(offset + seq_len + 100)
        
        # Seleciona cache relevante
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Aplica rotação 2D."""
        # x: (batch, seq, heads, dim)
        # cos, sin: (seq, dim)
        
        # Reshape para broadcast
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # Rotação
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        
        # [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]],
        ], dim=-1)
        
        return rotated

class RotarySelfAttention(nn.Module):
    """Self-attention com RoPE."""
    
    def __init__(self, d_model: int, n_heads: int = 8, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Projeções
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim) for t in qkv]
        
        # Aplica RoPE a Q e K
        q = self.rope(q)
        k = self.rope(k)
        
        # Reshape para atenção
        q = q.transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Máscara causal
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Atenção
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        
        # Output
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

def compare_position_embeddings():
    """Compara embeddings posicionais tradicionais vs RoPE."""
    
    print("Comparação: Position Embeddings vs RoPE")
    print("="*60)
    
    d_model = 256
    n_heads = 8
    seq_len = 128
    
    # Standard
    standard_pe = nn.Embedding(seq_len, d_model)
    
    # RoPE
    rope = RotaryPositionEmbedding(d_model // n_heads, seq_len)
    
    print(f"\nPosition Embeddings:")
    print(f"  Parâmetros: {seq_len * d_model:,}")
    print(f"  Geralização: Limitada ao seq_len treinado")
    
    print(f"\nRoPE:")
    print(f"  Parâmetros: 0 (não aprendido)")
    print(f"  Generalização: Extrapola para sequências maiores")
    print(f"  Codificação: Relativa (mais natural)")
    
    # Teste de extrapolação
    print("\n" + "="*60)
    print("Teste de Extrapolação:")
    print("="*60)
    
    # Durante treino: seq_len = 128
    # Durante inferência: seq_len = 256
    
    # Standard PE: Falha (índices fora do range)
    # RoPE: Funciona (calcula frequências para qualquer posição)
    
    long_seq = torch.randn(1, 256, d_model)
    
    print("\nStandard PE (seq_len=128):")
    print("  Inferência com seq_len=256: ERRO (out of bounds)")
    
    print("\nRoPE:")
    print("  Inferência com seq_len=256: OK (calcula dinamicamente)")

if __name__ == "__main__":
    compare_position_embeddings()
```

Execute:

```bash
python exercises/solution_04.py
```

---

## Resumo das Melhorias

| Melhoria | Dificuldade | Impacto | Comando |
|----------|-------------|---------|---------|
| Adicionar jogador | Fácil | Médio | `python download_players.py --players karpov` |
| Tuning hiperparâmetros | Médio | Alto | `python hyperparameter_search.py` |
| Avaliação Stockfish | Médio | Alto | `python evaluate_with_stockfish.py` |
| Implementar RoPE | Difícil | Alto | Modificar arquitetura |

---

## Links Relacionados

- [[exercicios/exercicio-04-melhorias|Enunciado do exercício]]
- [[ChessLM]] - Visão geral
- [[03-Treinamento/finetune|Fine-tuning]]
