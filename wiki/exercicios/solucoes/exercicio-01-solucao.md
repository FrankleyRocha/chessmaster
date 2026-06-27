# Solução: Exercício 1 - Tokenizador

Soluções completas para o [[exercicios/exercicio-01-tokenizador|Exercício 1]].

## Tarefa 1: Explorar o Tokenizador

### Respostas

**1. Qual é o vocab_size?**

```python
tok = ChessTokenizer()
print(f"vocab_size: {tok.vocab_size}")  # ~50-52
```

O vocabulário tem aproximadamente 50 tokens (varia conforme duplicatas nos caracteres).

**2. O que acontece com caracteres desconhecidos?**

```python
ids = tok.encode("xyz!@#")
print(ids)  # [1, 1, 1, 1, 1, 1] (todos UNK, id=1)
```

Caracteres não reconhecidos são mapeados para `<UNK>` (token ID 1).

**3. Quantos tokens na string de exemplo?**

```python
text = "1. e4 e5 2. Nf3 Nc6"
ids = tok.encode(text)
print(f"{len(ids)} tokens")  # 18 tokens
```

---

## Tarefa 2: Adicionar Novos Caracteres

```python
# model/tokenizer.py

CHESS_CHARS = (
    "abcdefgh"
    "12345678"
    "RNBQK"
    "x+=#+."
    "O-"
    "0123456789"
    " \n/"
    "*"
    "!?"  # ADICIONADO
)

# Teste
tok = ChessTokenizer()
print(f"Novo vocab_size: {tok.vocab_size}")  # 52

# Verificar
assert tok.encode("!")[0] != tok.unk_id  # '!' agora é reconhecido
assert tok.encode("?")[0] != tok.unk_id  # '?' também

print("✓ Caracteres adicionados com sucesso")
```

---

## Tarefa 3: Implementar Padding

```python
import torch

def batch_encode(
    tok: ChessTokenizer,
    texts: list[str],
    max_length: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokeniza batch com padding."""
    batch_size = len(texts)
    
    # Inicializa com PAD
    input_ids = torch.full((batch_size, max_length), tok.pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    
    for i, text in enumerate(texts):
        # Tokeniza
        ids = tok.encode(text)
        
        # Trunca se necessário
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        # Preenche
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids)
        attention_mask[i, :length] = 1
    
    return input_ids, attention_mask

# Teste
tok = ChessTokenizer()
texts = ["1. e4 e5", "1. d4", "1. Nf3 Nc6 Bb5"]
input_ids, mask = batch_encode(tok, texts, max_length=15)

print("input_ids:")
print(input_ids)
print("\nattention_mask:")
print(mask)

# Verificar
assert input_ids.shape == (3, 15)
assert mask.shape == (3, 15)
assert (mask.sum(dim=1) == torch.tensor([7., 4., 10.])).all()
```

---

## Tarefa 4: Frequência de Tokens

```python
from collections import Counter
import matplotlib.pyplot as plt

def analyze_token_frequency(tok: ChessTokenizer, text: str) -> dict[int, int]:
    """Analisa frequência de tokens."""
    ids = tok.encode(text)
    return dict(Counter(ids))

def plot_token_freq(tok: ChessTokenizer, text: str, top_n: int = 15):
    """Plota gráfico de frequência."""
    freq = analyze_token_frequency(tok, text)
    
    # Top N
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    tokens = [tok.decode([t]) for t, _ in sorted_freq]
    counts = [c for _, c in sorted_freq]
    
    plt.figure(figsize=(10, 5))
    plt.bar(tokens, counts)
    plt.xlabel("Token")
    plt.ylabel("Frequência")
    plt.title(f"Top {top_n} Tokens Mais Frequentes")
    plt.tight_layout()
    plt.savefig("token_freq.png")
    plt.show()

# Análise do dataset
tok = ChessTokenizer()

with open("data/pretrain.txt", "r", encoding="utf-8") as f:
    text = f.read()

freq = analyze_token_frequency(tok, text)

print("\nTop 10 tokens:")
print("Rank | Token | Contagem")
print("-" * 30)

for rank, (token_id, count) in enumerate(sorted(freq.items(), key=lambda x: -x[1])[:10], 1):
    token = tok.decode([token_id])
    print(f"{rank:4d} | {repr(token):6s} | {count:,}")

# plot_token_freq(tok, text)
```

Output típico:

```
Top 10 tokens:
Rank | Token | Contagem
------------------------------
   1 | ' '   | 298,543
   2 | '1'   | 142,231
   3 | '.'   | 98,453
   4 | 'e'   | 81,234
   5 | '4'   | 67,891
   6 | '2'   | 61,234
   7 | '5'   | 58,123
   8 | '3'   | 54,678
   9 | 'N'   | 43,219
  10 | 'd'   | 41,987
```

---

## Código Completo

```python
#!/usr/bin/env python
"""Soluções completas do Exercício 1."""

import torch
from collections import Counter
import matplotlib.pyplot as plt

from model.tokenizer import ChessTokenizer

def main():
    print("="*60)
    print("Exercício 1: Tokenizador - Soluções")
    print("="*60)
    
    # Tarefa 1
    print("\n[Tarefa 1] Explorando o tokenizador...")
    tok = ChessTokenizer()
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  UNK test: {tok.encode('xyz')}")
    print(f"  Token count: {len(tok.encode('1. e4 e5 2. Nf3 Nc6'))}")
    
    # Tarefa 2
    print("\n[Tarefa 2] Verificando caracteres...")
    # (requer modificação do tokenizer.py)
    print("  Verifique se !? foram adicionados ao CHESS_CHARS")
    
    # Tarefa 3
    print("\n[Tarefa 3] Batch encode com padding...")
    texts = ["1. e4", "1. d4 d5 2. c4", "1. Nf3"]
    input_ids, mask = batch_encode(tok, texts, max_length=10)
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  mask shape: {mask.shape}")
    
    # Tarefa 4
    print("\n[Tarefa 4] Análise de frequência...")
    try:
        with open("data/pretrain.txt", "r") as f:
            text = f.read()
        freq = analyze_token_frequency(tok, text)
        print(f"  Tokens únicos: {len(freq)}")
        print(f"  Total tokens: {sum(freq.values()):,}")
    except FileNotFoundError:
        print("  Arquivo data/pretrain.txt não encontrado")
    
    print("\n" + "="*60)
    print("Soluções completas!")
    print("="*60)

if __name__ == "__main__":
    main()
```

Execute com:

```bash
cd /home/frank/projetos/chessmaster/chesslm
python exercises/solution_01.py
```
