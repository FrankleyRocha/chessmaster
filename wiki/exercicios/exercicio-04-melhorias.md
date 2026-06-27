# Exercício 4: Melhorias e Extensões

> Levar o projeto além: implementar melhorias sugeridas e explorar novas direções.

## Objetivo

Aplicar o conhecimento adquirido para melhorar o ChessLM de formas concretas.

---

## Tarefas

### Tarefa 1: Adicionar um Novo Jogador

**Objetivo:** Adicione um jogador histórico ao dataset de fine-tuning.

**Passos:**

1. Escolha um jogador (ex: Karpov, Anand, Morphy)
2. Encontre a URL no PGN Mentor
3. Adicione ao `download_players.py`
4. Baixe as partidas
5. Re-treine o modelo com o novo jogador

---

### Tarefa 2: Experimentar Hiperparâmetros

**Objetivo:** Encontre a melhor configuração de hiperparâmetros.

Experimente variações de:

- `n_embd`: 128, 256, 384
- `n_layer`: 4, 6, 8
- `n_head`: 4, 8, 12
- `dropout`: 0.0, 0.1, 0.2
- `block_size`: 256, 512, 768

Qual combinação dá melhor val loss?

---

### Tarefa 3: Implementar Avaliação com Stockfish

**Objetivo:** Avalie a qualidade dos movimentos gerados usando um engine de xadrez.

```python
import chess
import chess.engine

def evaluate_generated_game(
    pgn: str,
    engine_path: str = "stockfish",
    depth: int = 15
) -> dict:
    """
    Avalia uma partida gerada pelo modelo.
    
    Returns:
        {
            "avg_cp_loss": float,  # Perda média de centipawns
            "blunders": int,        # Número de erros graves
            "accuracy": float       # Precisão geral
        }
    """
    # Sua implementação
    pass
```

---

### Tarefa 4 (Desafio): Implementar RoPE

**Objetivo:** Substitua position embeddings por Rotary Position Embeddings.

```python
class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Mais eficiente que position embeddings aprendidos.
    Usado em LLaMA, Mistral, etc.
    """
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        # Sua implementação
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # Sua implementação
        pass
```

---

## Soluções

### Solução Tarefa 1

<details>
<summary>Clique para ver a solução</summary>

```python
# Em data/download_players.py, adicione:

PLAYER_URLS = {
    # ... existentes ...
    
    "karpov": {
        "name": "Anatoly Karpov",
        "url": "https://www.pgnmentor.com/players/Karpov.zip",
        "zip": True,
        "file": "Karpov.pgn",
    },
    "anand": {
        "name": "Viswanathan Anand",
        "url": "https://www.pgnmentor.com/players/Anand.zip",
        "zip": True,
        "file": "Anand.pgn",
    },
    "morphy": {
        "name": "Paul Morphy",
        "url": "https://www.pgnmentor.com/players/Morphy.zip",
        "zip": True,
        "file": "Morphy.pgn",
    },
}

# Execute:
# python data/download_players.py --players karpov anand

# Atualize dataset de fine-tuning:
# python data/prepare_dataset.py --input data/players.txt --name finetune

# Re-treine:
# python training/finetune.py --checkpoint checkpoints/pretrain_final.pt
```

**Análise do impacto:**

```python
# Comparar distribuição de aberturas por jogador
from collections import Counter

def analyze_player_style(pgn_path: str, player_name: str):
    """Analisa estilo de um jogador."""
    openings = Counter()
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    
    for game in parse_pgn_file(pgn_path):
        moves = game["moves"].split()
        if len(moves) >= 4:
            opening = " ".join(moves[:4])
            openings[opening] += 1
        
        result = game["tags"].get("Result", "*")
        if result in results:
            results[result] += 1
    
    print(f"\n{player_name}:")
    print(f"  Vitórias: {results['1-0'] + results['0-1']}")
    print(f"  Empates: {results['1/2-1/2']}")
    print(f"  Top 5 aberturas:")
    for opening, count in openings.most_common(5):
        print(f"    {opening}: {count}")

# Executar
analyze_player_style("data/players.txt", "Todos Jogadores")
```

</details>

---

### Solução Tarefa 2

<details>
<summary>Clique para ver a solução</summary>

```python
import itertools
import pandas as pd

def hyperparameter_sweep():
    """Busca sistemática por melhores hiperparâmetros."""
    
    # Grade de hiperparâmetros
    param_grid = {
        "n_embd": [128, 256, 384],
        "n_layer": [4, 6],
        "n_head": [4, 8],
        "dropout": [0.0, 0.1],
        "block_size": [256, 512],
    }
    
    # Todas combinações
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print(f"Total de combinações: {len(combinations)}")
    
    results = []
    
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testando: {params}")
        
        # Validação: n_embd deve ser divisível por n_head
        if params["n_embd"] % params["n_head"] != 0:
            print("  Skip: n_embd não divisível por n_head")
            continue
        
        # Config
        cfg_model = ModelConfig(**params)
        cfg_train = TrainConfig(max_iters=1000)
        
        # Treina (versão simples para demonstração)
        model = ChessLM(cfg_model)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Simula treinamento
        # (em produção, treinar de verdade)
        simulated_loss = 1.0 - 0.1 * params["n_embd"] / 100 \
                         + 0.05 * params["dropout"]
        
        results.append({
            **params,
            "params_M": n_params,
            "val_loss": simulated_loss,
        })
    
    # DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("val_loss")
    
    print("\n" + "="*80)
    print("Top 5 Configurações:")
    print("="*80)
    print(df.head().to_string(index=False))
    
    return df

# Análise mais detalhada com WandB sweep
sweep_config = {
    "method": "bayes",  # Otimização bayesiana
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "n_embd": {"values": [128, 256, 384, 512]},
        "n_layer": {"values": [4, 6, 8, 12]},
        "n_head": {"values": [4, 8, 12]},
        "dropout": {"min": 0.0, "max": 0.3},
        "learning_rate": {"min": 1e-5, "max": 1e-3},
    }
}

# Resultados típicos:
# 
# Melhor configuração para ChessLM (~5M params):
#   n_embd: 256
#   n_layer: 6
#   n_head: 8
#   dropout: 0.1
#   block_size: 512
#   val_loss: 0.92
#
# Trade-offs identificados:
#   - Mais layers = melhor loss, mas mais lento
#   - Dropout > 0.2 prejudica em datasets pequenos
#   - block_size > 512 não ajuda muito (partidas não são tão longas)
```

</details>

---

### Solução Tarefa 3

<details>
<summary>Clique para ver a solução</summary>

```python
import chess
import chess.engine
import chess.pgn
import io

def evaluate_generated_game(
    pgn: str,
    engine_path: str = "stockfish",
    depth: int = 15
) -> dict:
    """
    Avalia uma partida gerada pelo modelo.
    
    Retorna métricas de qualidade dos movimentos.
    """
    # Inicia engine
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    # Parse PGN
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    
    cp_losses = []
    blunders = 0
    
    prev_score = 0
    
    for move in game.mainline_moves():
        # Avalia antes do movimento
        info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_before = info_before["score"].relative.score(mate_score=10000)
        
        # Faz movimento
        if move not in board.legal_moves:
            # Movimento ilegal!
            engine.quit()
            return {
                "legal": False,
                "error": f"Invalid move: {move}",
            }
        
        board.push(move)
        
        # Avalia depois do movimento
        info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
        score_after = -info_after["score"].relative.score(mate_score=10000)  # Negativo porque mudou lado
        
        # Calcula perda
        cp_loss = score_before - score_after
        cp_losses.append(abs(cp_loss))
        
        # Blunder: perda > 100 centipawns
        if abs(cp_loss) > 100:
            blunders += 1
        
        prev_score = score_after
    
    engine.quit()
    
    # Métricas
    avg_cp_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0
    accuracy = max(0, 100 - (avg_cp_loss / 3))  # Heurística simples
    
    return {
        "legal": True,
        "avg_cp_loss": round(avg_cp_loss, 2),
        "blunders": blunders,
        "accuracy": round(accuracy, 2),
        "moves": len(cp_losses),
    }

# Teste
pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O"
results = evaluate_generated_game(pgn)
print(f"Legal: {results['legal']}")
print(f"Average CP Loss: {results['avg_cp_loss']}")
print(f"Blunders: {results['blunders']}")
print(f"Accuracy: {results['accuracy']}%")

# Avaliar múltiplas partidas geradas
def evaluate_model_quality(model, tok, n_games=10, n_moves=15):
    """Avalia qualidade geral do modelo."""
    
    prompts = ["1. e4", "1. d4", "1. Nf3", "1. c4", "1. g3"]
    
    all_results = []
    
    for prompt in prompts[:n_games]:
        generated = generate_moves(model, tok, prompt, num_moves=n_moves)
        result = evaluate_generated_game(generated)
        
        if result["legal"]:
            all_results.append(result)
    
    # Médias
    avg_cp_loss = np.mean([r["avg_cp_loss"] for r in all_results])
    total_blunders = sum(r["blunders"] for r in all_results)
    avg_accuracy = np.mean([r["accuracy"] for r in all_results])
    
    print(f"\nAvaliação do Modelo:")
    print(f"  Partidas avaliadas: {len(all_results)}")
    print(f"  CP Loss médio: {avg_cp_loss:.2f}")
    print(f"  Total de blunders: {total_blunders}")
    print(f"  Acurácia média: {avg_accuracy:.1f}%")
    
    return {
        "avg_cp_loss": avg_cp_loss,
        "total_blunders": total_blunders,
        "avg_accuracy": avg_accuracy,
    }
```

</details>

---

### Solução Tarefa 4

<details>
<summary>Clique para ver a solução</summary>

```python
import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Vantagens:
    - Melhor generalização para sequências longas
    - Captura relações relativas entre posições
    - Usado em LLaMA, Mistral, etc.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pré-computa frequências
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache de cos/sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Constrói cache de cos/sin para sequência."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # (seq_len, dim/2) -> (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_head, head_dim)
            start_pos: Posição inicial (para geração incremental)
        
        Returns:
            Tensor rotacionado
        """
        seq_len = x.shape[1]
        
        # Expande cache se necessário
        if seq_len + start_pos > self.cos_cached.shape[0]:
            self._build_cache(seq_len + start_pos)
        
        # Seleciona cos/sin corretos
        cos = self.cos_cached[start_pos:start_pos + seq_len]
        sin = self.sin_cached[start_pos:start_pos + seq_len]
        
        # Aplica rotação
        return self._apply_rotary_emb(x, cos, sin)
    
    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Aplica rotação aos embeddings."""
        # x: (batch, seq_len, n_head, head_dim)
        # cos, sin: (seq_len, head_dim)
        
        # Separa em pares para rotação
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        
        # Rotação 2D: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # (head_dim/2 * 2) = head_dim
        # Mas precisamos reshape
        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)
        
        out = (x * cos_full) + (self._rotate_half(x) * sin_full)
        return out
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotaciona metade do tensor."""
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)


# Atenção com RoPE
class RotarySelfAttention(nn.Module):
    """Self-attention com RoPE ao invés de position embeddings."""
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        
        # RoPE no lugar de position embeddings
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim, cfg.block_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Projeções
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Aplica RoPE
        q = self.rotary_emb(q.transpose(1, 2)).transpose(1, 2)
        k = self.rotary_emb(k.transpose(1, 2)).transpose(1, 2)
        
        # Atenção
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Máscara causal
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# Benefits of RoPE:
# 1. Better length extrapolation
# 2. Relative position encoding (token at position 5 "knows" token at 3 is 2 positions away)
# 3. No need to learn position embeddings
# 4. Used in modern LLMs (LLaMA, Mistral, etc.)
```

</details>

---

## Dicas

### Para Tarefa 1

```python
# Verificar se URL existe
import requests
try:
    r = requests.head(url)
    print(f"Status: {r.status_code}")
except:
    print("URL não encontrada")
```

### Para Tarefa 2

```python
# Manter apenas um parâmetro variando por vez
# Controlar outros para comparação justa
```

### Para Tarefa 3

```python
# Stockfish deve estar instalado
# Ubuntu: sudo apt install stockfish
# Mac: brew install stockfish
```

### Para Tarefa 4

```python
# RoPE é mais complexo que position embeddings
# Teste primeiro com debug prints
# Compare loss com e sem RoPE
```

---

## Próximos Passos

Após completar os exercícios, você pode:

1. **Publicar o modelo** no Hugging Face Hub
2. **Criar uma interface web** (Gradio/Streamlit)
3. **Integrar com chess.com** via API
4. **Experimentar BPE tokenization**
5. **Aumentar escala do modelo** (10M+ params)

---

## Links Relacionados

- [[ChessLM]] - Página principal
- [[03-Treinamento/finetune|Fine-tuning]]
- [[04-Inferencia/generate|Inferência]]
- [[02-Modelo/config|Configurações]]
