# Solução: Exercício 2 - Mecanismo de Atenção

Soluções completas para o [[exercicios/exercicio-02-atencao|Exercício 2]].

## Tarefa 1: Visualizar Matriz de Atenção

```python
#!/usr/bin/env python
"""Visualização de matrizes de atenção."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import ModelConfig
from model.model import ChessLM
from model.tokenizer import ChessTokenizer

def extract_attention(
    model: ChessLM,
    tok: ChessTokenizer,
    text: str,
    layer: int = 0,
    head: int = 0
) -> np.ndarray:
    """Extrai matriz de atenção do modelo."""
    
    model.eval()
    
    with torch.no_grad():
        ids = tok.encode(text)
        x = torch.tensor([ids])
        B, T = x.shape
        pos = torch.arange(0, T)
        
        # Embeddings
        tok_emb = model.transformer.wte(x)
        pos_emb = model.transformer.wpe(pos)
        x = model.transformer.drop(tok_emb + pos_emb)
        
        # Passa pelos blocos até o layer desejado
        for i in range(layer):
            x = model.transformer.h[i](x)
        
        # Extrai atenção do layer
        block = model.transformer.h[layer]
        ln_x = block.ln1(x)
        
        # Atenção manual
        B, T, C = ln_x.shape
        qkv = block.attn.c_attn(ln_x)
        q, k, v = qkv.split(C, dim=2)
        
        n_head = block.attn.n_head
        head_dim = C // n_head
        
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        
        # Score
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Máscara causal
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        
        return att[0, head].cpu().numpy()

def plot_attention(
    attention: np.ndarray,
    tokens: list[str],
    title: str = "Attention Matrix"
):
    """Plota heatmap de atenção."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )
    
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig("attention_matrix.png", dpi=150)
    plt.show()

def main():
    print("Visualizando Matriz de Atenção")
    print("="*50)
    
    # Carrega modelo
    tok = ChessTokenizer()
    cfg = ModelConfig()
    model = ChessLM(cfg)
    
    # Tenta carregar checkpoint
    ckpt_path = Path("checkpoints/pretrain_best.pt")
    if ckpt_path.exists():
        print(f"Carregando {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    else:
        print("Usando modelo sem treino (random weights)")
    
    # Texto para análise
    text = "1. e4 e5 2. Nf3 Nc6 3. Bb5"
    tokens = list(text)
    
    print(f"\nTexto: {text}")
    print(f"Tokens: {len(tokens)}")
    
    # Extrai atenção
    for layer in [0, 2, 5]:
        attention = extract_attention(model, tok, text, layer=layer, head=0)
        print(f"\nLayer {layer}, Head 0:")
        print(f"  Shape: {attention.shape}")
        print(f"  Max attention: {attention.max():.4f}")
        
        # Plot
        try:
            plot_attention(attention, tokens, title=f"Layer {layer}, Head 0")
        except Exception as e:
            print(f"  Could not plot: {e}")

if __name__ == "__main__":
    main()
```

---

## Tarefa 2: Efeito do Número de Cabeças

```python
#!/usr/bin/env python
"""Compara modelos com diferentes números de cabeças."""

import torch
import time
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import ModelConfig, TrainConfig
from model.model import ChessLM

def count_parameters(model: torch.nn.Module) -> int:
    """Conta parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_heads(n_heads_list: list[int] = [1, 2, 4, 8]):
    """Benchmarks diferentes números de cabeças."""
    
    results = []
    
    for n_head in n_heads_list:
        print(f"\n{'='*50}")
        print(f"Testando n_head={n_head}")
        print(f"{'='*50}")
        
        # Config
        cfg = ModelConfig(n_head=n_head)
        
        # Validação
        if cfg.n_embd % n_head != 0:
            print(f"  Skip: n_embd ({cfg.n_embd}) não divisível por n_head ({n_head})")
            continue
        
        # Modelo
        model = ChessLM(cfg)
        n_params = count_parameters(model)
        
        # Benchmark forward pass
        batch_size = 8
        seq_len = 128
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        # Warmup
        _ = model(x)
        
        # Medir tempo
        n_runs = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x)
        elapsed = time.time() - start
        
        results.append({
            "n_head": n_head,
            "params_M": n_params / 1e6,
            "time_ms": elapsed / n_runs * 1000,
        })
        
        print(f"  Parâmetros: {n_params/1e6:.2f}M")
        print(f"  Tempo/forward: {elapsed/n_runs*1000:.2f}ms")
    
    return pd.DataFrame(results)

def main():
    print("Comparação de Cabeças de Atenção")
    print("="*50)
    
    df = benchmark_heads()
    
    print("\n" + "="*50)
    print("Resultados:")
    print("="*50)
    print(df.to_string(index=False))
    
    # Salvar
    df.to_csv("attention_heads_comparison.csv", index=False)
    print("\nSalvo em: attention_heads_comparison.csv")

if __name__ == "__main__":
    main()
```

---

## Tarefa 3: Atenção Simplificada

```python
#!/usr/bin/env python
"""Implementação simplificada de atenção."""

import torch
import math

def simple_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implementação simplificada de self-attention.
    
    Args:
        Q: Queries (batch, seq_len, d_k)
        K: Keys (batch, seq_len, d_k)
        V: Values (batch, seq_len, d_v)
        mask: Máscara opcional (seq_len, seq_len)
    
    Returns:
        output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # 1. Scores: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. Escala
    scores = scores / math.sqrt(d_k)
    
    # 3. Máscara
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 5. Output: att @ V
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

def test_simple_attention():
    """Testa implementação."""
    print("Testando Simple Attention")
    print("="*50)
    
    # Setup
    batch_size = 2
    seq_len = 5
    d_k = d_v = 8
    
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    # Máscara causal
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # Forward
    output, weights = simple_attention(Q, K, V, mask=causal_mask)
    
    print(f"Input shape: ({batch_size}, {seq_len}, {d_k})")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # Verifica propriedades
    print("\nVerificações:")
    
    # 1. Causalidade
    print("1. Causalidade (triangular inferior):")
    print(weights[0].numpy().round(2))
    
    # 2. Soma = 1
    row_sums = weights.sum(dim=-1)
    print(f"\n2. Soma por linha = 1: {torch.allclose(row_sums, torch.ones_like(row_sums))}")
    
    # 3. Valores não negativos
    print(f"3. Todos valores >= 0: {(weights >= 0).all().item()}")
    
    return output, weights

if __name__ == "__main__":
    test_simple_attention()
```

---

## Tarefa 4: Sliding Window Attention

```python
#!/usr/bin/env python
"""Implementação de Sliding Window Attention."""

import torch
import torch.nn as nn
import math
import time

class SlidingWindowAttention(nn.Module):
    """
    Atenção com janela deslizante.
    
    Cada token só atende aos `window_size` tokens anteriores.
    Complexidade: O(n * window_size) ao invés de O(n²).
    """
    
    def __init__(self, d_model: int, n_head: int = 8, window_size: int = 128):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.window_size = window_size
        self.head_dim = d_model // n_head
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape
        
        # Projeções
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Máscara de janela deslizante
        positions = torch.arange(T, device=x.device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        window_mask = (distance >= 0) & (distance < self.window_size)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        
        # Aplica máscara
        scores = scores.masked_fill(~window_mask, float('-inf'))
        
        # Atenção
        attn = torch.softmax(scores, dim=-1)
        
        # Remove NaN de linhas totalmente mascaradas (não deve acontecer com janela correta)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        # Output
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_head={self.n_head}, window_size={self.window_size}"

def benchmark_sliding_window():
    """Compara atenção padrão vs sliding window."""
    
    d_model = 256
    n_head = 8
    window_size = 64
    
    seq_lengths = [128, 256, 512, 1024]
    
    print("Benchmark: Standard vs Sliding Window Attention")
    print("="*60)
    
    # Modelos
    standard_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
    sliding_attn = SlidingWindowAttention(d_model, n_head, window_size)
    
    results = []
    
    for seq_len in seq_lengths:
        batch_size = 4
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Standard
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = standard_attn(x, x, x, need_weights=False)
        standard_time = (time.time() - start) / 10 * 1000
        
        # Sliding window
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = sliding_attn(x)
        sliding_time = (time.time() - start) / 10 * 1000
        
        results.append({
            "seq_len": seq_len,
            "standard_ms": standard_time,
            "sliding_ms": sliding_time,
            "speedup": standard_time / sliding_time,
        })
    
    print(f"\n{'Seq Len':<10} | {'Standard':<12} | {'Sliding':<12} | {'Speedup'}")
    print("-" * 55)
    for r in results:
        print(f"{r['seq_len']:<10} | {r['standard_ms']:>10.2f}ms | {r['sliding_ms']:>10.2f}ms | {r['speedup']:.2f}x")

if __name__ == "__main__":
    benchmark_sliding_window()
```

Execute:

```bash
python exercises/solution_02.py
```
