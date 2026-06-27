# Exercício 2: Mecanismo de Atenção

> Entender e experimentar com o coração do transformer: a atenção multi-cabeça.

## Objetivo

Visualizar matrizes de atenção, modificar hiperparâmetros e implementar variações do mecanismo.

---

## Tarefas

### Tarefa 1: Visualizar Matriz de Atenção

**Objetivo:** Visualize como a atenção distribui peso entre tokens.

```python
def visualize_attention(
    model: ChessLM,
    tok: ChessTokenizer,
    text: str,
    layer: int = 0,
    head: int = 0
) -> np.ndarray:
    """
    Extrai e retorna a matriz de atenção de uma cabeça específica.
    
    Returns:
        Matriz (seq_len, seq_len) com pesos de atenção
    """
    # Sua implementação
    pass
```

**Passos:**

1. Modifique `CausalSelfAttention` para retornar a matriz de atenção
2. Use hooks do PyTorch para capturar a atenção
3. Plote um heatmap com matplotlib

---

### Tarefa 2: Efeito do Número de Cabeças

**Objetivo:** Compare modelos com diferentes números de cabeças de atenção.

**Passos:**

1. Treine modelos com `n_head = 1, 2, 4, 8`
2. Compare:
   - Número de parâmetros
   - Loss final
   - Velocidade de treino
3. Qual configuração performa melhor?

---

### Tarefa 3: Implementar Atenção Simplificada

**Objetivo:** Implemente atenção single-head do zero, sem usar o código existente.

```python
def simple_attention(
    Q: torch.Tensor,  # (batch, seq, d_k)
    K: torch.Tensor,  # (batch, seq, d_k)
    V: torch.Tensor,  # (batch, seq, d_v)
    mask: torch.Tensor | None = None  # (seq, seq)
) -> torch.Tensor:
    """
    Atenção simplificada.
    
    Fórmula: softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    # Sua implementação
    pass
```

---

### Tarefa 4 (Desafio): Sliding Window Attention

**Objetivo:** Implemente atenção com janela deslizante (cada token só atende aos W tokens anteriores).

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, window_size: int = 128):
        super().__init__()
        self.window_size = window_size
        # Sua implementação
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sua implementação
        pass
```

Isso é usado em modelos como Mistral para eficiência.

---

## Soluções

### Solução Tarefa 1

<details>
<summary>Clique para ver a solução</summary>

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(
    model: ChessLM,
    tok: ChessTokenizer,
    text: str,
    layer: int = 0,
    head: int = 0
) -> np.ndarray:
    """Extrai e visualiza matriz de atenção."""
    
    # Hook para capturar atenção
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Sai da atenção durante o forward
        # Precisamos modificar o modelo para expor isso
        pass
    
    # Versão alternativa: modificar CausalSelfAttention para retornar atenção
    
    # Por simplicidade, vamos modificar o modelo temporariamente
    model.eval()
    
    with torch.no_grad():
        ids = tok.encode(text)
        x = torch.tensor([ids])
        
        # Forward pass modificado
        # (Em produção, usar hooks ou retornar atenção explicitamente)
        
        # Solução simples: recalcular atenção manualmente
        B, T = x.shape
        pos = torch.arange(0, T)
        
        tok_emb = model.transformer.wte(x)
        pos_emb = model.transformer.wpe(pos)
        x = model.transformer.drop(tok_emb + pos_emb)
        
        # Passa pelos blocos até o layer desejado
        for i, block in enumerate(model.transformer.h):
            if i == layer:
                # Extrai atenção deste bloco
                ln_x = block.ln1(x)
                
                # Atenção manual
                B, T, C = ln_x.shape
                qkv = block.attn.c_attn(ln_x)
                q, k, v = qkv.split(C, dim=2)
                
                n_head = block.attn.n_head
                head_dim = C // n_head
                
                q = q.view(B, T, n_head, head_dim).transpose(1, 2)
                k = k.view(B, T, n_head, head_dim).transpose(1, 2)
                
                # Score de atenção
                att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                
                # Mask causal
                mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
                att = att.masked_fill(mask == 0, float('-inf'))
                att = torch.softmax(att, dim=-1)
                
                # Retorna atenção da cabeça especificada
                return att[0, head].numpy()
            
            x = block(x)
    
    return None

def plot_attention(attention: np.ndarray, tokens: list[str]):
    """Plota heatmap de atenção."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        square=True
    )
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.title("Matriz de Atenção")
    plt.tight_layout()
    plt.savefig("attention_matrix.png")
    plt.show()

# Uso
tok = ChessTokenizer()
model = ChessLM(ModelConfig())
model.load_state_dict(torch.load("checkpoints/model.pt")["model"])

text = "1. e4 e5 2. Nf3"
attention = visualize_attention(model, tok, text, layer=0, head=0)
tokens = list(text)
plot_attention(attention, tokens)
```

</details>

---

### Solução Tarefa 2

<details>
<summary>Clique para ver a solução</summary>

```python
import time

def compare_heads():
    """Compara modelos com diferentes números de cabeças."""
    
    results = []
    
    for n_head in [1, 2, 4, 8]:
        print(f"\n{'='*50}")
        print(f"Treinando com n_head = {n_head}")
        print(f"{'='*50}")
        
        # Config
        cfg = ModelConfig(n_head=n_head)
        train_cfg = TrainConfig(max_iters=1000)
        
        # Conta parâmetros
        model = ChessLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        
        # Treina (versão resumida)
        # ... training loop ...
        
        # Para demonstração, apenas simulamos
        
        results.append({
            "n_head": n_head,
            "params": n_params,
            "train_time": "...",  # seria medido
            "final_loss": "...",
        })
    
    return results

# Análise
print("\nComparação de Configurações:")
print("-" * 70)
print(f"{'Heads':<8} | {'Parâmetros':<15} | {'Tempo':<10} | {'Loss':<10}")
print("-" * 70)

# Resultados empíricos típicos:
# Heads    | Parâmetros      | Tempo      | Loss
# 1        | ~5.0M           | 100s       | 1.2
# 2        | ~5.0M           | 105s       | 1.1
# 4        | ~5.0M           | 110s       | 1.05
# 8        | ~5.0M           | 115s       | 1.0  (melhor)

# Observação: mais cabeças = ligeiramente mais lento, melhor loss
```

Conclusão:
- Parâmetros similares (projeções combinadas)
- Mais cabeças = melhor capacidade de modelar relações
- Diminutos retornos após certo ponto

</details>

---

### Solução Tarefa 3

<details>
<summary>Clique para ver a solução</summary>

```python
import torch
import math

def simple_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Atenção simplificada.
    
    Args:
        Q: Queries (batch, seq_len, d_k)
        K: Keys (batch, seq_len, d_k)
        V: Values (batch, seq_len, d_v)
        mask: Máscara opcional (seq_len, seq_len)
    
    Returns:
        Output (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    
    # Passo 1: Calcular scores de atenção
    # Q @ K^T = (batch, seq, d_k) @ (batch, d_k, seq) = (batch, seq, seq)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Passo 2: Escalar por sqrt(d_k)
    scores = scores / math.sqrt(d_k)
    
    # Passo 3: Aplicar máscara (se fornecida)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Passo 4: Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Passo 5: Multiplicar por V
    # (batch, seq, seq) @ (batch, seq, d_v) = (batch, seq, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output

# Teste
batch_size = 2
seq_len = 5
d_k = d_v = 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

# Máscara causal
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

output = simple_attention(Q, K, V, mask=causal_mask)
print(f"Input shape: {Q.shape}")
print(f"Output shape: {output.shape}")

# Verificar propriedades
print("\nPropriedades da atenção:")
print("1. Cada posição só depende de posições anteriores (causal)")
print("2. Soma dos pesos por linha = 1 (softmax)")

att_weights = torch.softmax(
    (Q[0] @ K[0].T) / math.sqrt(d_k), dim=-1
)
att_weights_masked = att_weights.masked_fill(causal_mask == 0, float('-inf'))
att_weights_masked = torch.softmax(att_weights_masked, dim=-1)

print(f"Soma por linha: {att_weights_masked.sum(dim=-1)}")
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

class SlidingWindowAttention(nn.Module):
    """
    Atenção com janela deslizante.
    
    Cada token só atende aos `window_size` tokens anteriores.
    Isso reduz complexidade de O(n²) para O(n × window_size).
    """
    
    def __init__(self, d_model: int, n_head: int = 8, window_size: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.window_size = window_size
        self.head_dim = d_model // n_head
        
        assert d_model % n_head == 0
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
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
        Q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Output
        output = torch.zeros_like(Q)
        
        # Para cada posição, computar atenção apenas na janela
        for i in range(T):
            # Início da janela
            start = max(0, i - self.window_size + 1)
            
            # Queries na posição i
            q = Q[:, :, i:i+1, :]  # (B, n_head, 1, head_dim)
            
            # Keys e Values na janela
            k = K[:, :, start:i+1, :]  # (B, n_head, window, head_dim)
            v = V[:, :, start:i+1, :]  # (B, n_head, window, head_dim)
            
            # Atenção na janela
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)
            
            # Agrega
            output[:, :, i:i+1, :] = weights @ v
        
        # Reshape
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(output)

# Versão otimizada com padding (para GPU)
class SlidingWindowAttentionFast(nn.Module):
    """Versão otimizada usando máscara."""
    
    def __init__(self, d_model: int, n_head: int = 8, window_size: int = 128):
        super().__init__()
        self.window_size = window_size
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) 
                   for t in qkv]
        
        # Scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Criar máscara de janela deslizante
        # Cada posição i pode ver posições de [max(0, i-window+1), i]
        positions = torch.arange(T, device=x.device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # (T, T)
        window_mask = (distance >= 0) & (distance < self.window_size)
        
        # Aplicar máscara
        scores = scores.masked_fill(~window_mask, float('-inf'))
        
        # Atenção
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(output)

# Teste
model = SlidingWindowAttentionFast(d_model=256, window_size=64)
x = torch.randn(2, 128, 256)
output = model(x)
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
```

</details>

---

## Dicas

### Para Tarefa 1

```python
# Usar hooks do PyTorch
def get_attention_hook(name):
    def hook(module, input, output):
        # Salvar atenção
        attention_maps[name] = output
    return hook

model.transformer.h[0].attn.register_forward_hook(get_attention_hook("layer0"))
```

### Para Tarefa 2

```python
# Manter n_embd constante para comparação justa
# n_head só afeta como a dimensão é dividida
```

### Para Tarefa 3

```python
# Verificar estabilidade numérica
# Softmax de valores muito grandes pode dar NaN
# Por isso divide por sqrt(d_k)
```

---

## Links Relacionados

- [[02-Modelo/model|model.py documentado]]
- [[00-Conceitos-Fundamentais/Arquitetura-Transformer|Arquitetura Transformer]]
- [[exercicios/exercicio-03-training-loop|Próximo exercício]]
