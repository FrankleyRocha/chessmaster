# Exercício 1: Tokenizador

> Praticar a tokenização character-level e entender o pipeline de texto para IDs.

## Objetivo

Compreender como o tokenizador funciona, adicionar novas funcionalidades e implementar variações.

---

## Tarefas

### Tarefa 1: Explorar o Tokenizador

Execute o código abaixo e responda às perguntas:

```bash
cd /home/frank/projetos/chessmaster/chesslm
python -m model.tokenizer
```

**Perguntas:**

1. Qual é o `vocab_size` do tokenizador?
2. O que acontece quando você tenta tokenizar um caractere que não está no vocabulário?
3. Quantos tokens tem a string `"1. e4 e5 2. Nf3 Nc6"`?

---

### Tarefa 2: Adicionar Novos Caracteres

O tokenizador atual não suporta anotações de lance (`!`, `?`, `!!`, `??`, etc.).

**Objetivo:** Adicione suporte para os caracteres `!` e `?`.

**Passos:**

1. Abra o arquivo `model/tokenizer.py`
2. Modifique `CHESS_CHARS` para incluir `!?`
3. Verifique que encode/decode ainda funciona
4. Quantos tokens o vocabulário passou a ter?

---

### Tarefa 3: Implementar Padding

O tokenizador tem tokens especiais (`PAD`, `UNK`, `BOS`, `EOS`), mas não os usa ativamente.

**Objetivo:** Implemente uma função que recebe uma lista de strings e retorna tensores padded.

```python
def batch_encode(
    tok: ChessTokenizer,
    texts: list[str],
    max_length: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokeniza um batch de textos com padding.
    
    Args:
        tok: Tokenizador
        texts: Lista de strings PGN
        max_length: Tamanho máximo (trunca ou padding)
    
    Returns:
        input_ids: Tensor (batch, max_length)
        attention_mask: Tensor (batch, max_length) com 1 onde há tokens reais
    """
    # Sua implementação
    pass
```

---

### Tarefa 4 (Desafio): Frequência de Tokens

**Objetivo:** Implemente uma função que analisa a frequência de cada token no dataset.

```python
def analyze_token_frequency(
    tok: ChessTokenizer,
    text: str
) -> dict[int, int]:
    """
    Analisa frequência de cada token no texto.
    
    Returns:
        Dicionário {token_id: contagem}
    """
    # Sua implementação
    pass
```

Pergunta: Quais são os 10 tokens mais comuns em `data/pretrain.txt`?

---

## Soluções

### Solução Tarefa 1

<details>
<summary>Clique para ver a solução</summary>

```python
# Executando o script
tok = ChessTokenizer()
print(tok)
# ChessTokenizer(vocab_size=50)

# 1. vocab_size = ~50 (pode variar com caracteres duplicados)

# 2. Caractere desconhecido retorna UNK (id=1)
ids = tok.encode("xyz")  # 'x', 'y', 'z' não estão no vocabulário
print(ids)  # [1, 1, 1] (todos UNK)

# 3. Contando tokens
text = "1. e4 e5 2. Nf3 Nc6"
ids = tok.encode(text)
print(len(ids))  # 18 tokens
```

</details>

---

### Solução Tarefa 2

<details>
<summary>Clique para ver a solução</summary>

```python
# Em model/tokenizer.py, modifique:

CHESS_CHARS = (
    "abcdefgh"          # colunas
    "12345678"          # linhas
    "RNBQK"             # peças (maiúsculas)
    "x+=#+."            # captura, promoção, xeque, mate, ponto
    "O-"                # roque (letra O e hífen)
    "0123456789"        # dígitos para numeração dos movimentos
    " \n/"              # separadores
    "*"                 # resultado indeterminado
    "!?"                # <-- ADICIONADO: anotações de lance
)

# Novo vocab_size
tok = ChessTokenizer()
print(f"Novo vocab_size: {tok.vocab_size}")  # 52 (era 50)

# Teste
ids = tok.encode("1. e4!")
print(ids)  # Funciona, '!' tem seu próprio ID
print(tok.decode(ids))  # "1. e4!"
```

</details>

---

### Solução Tarefa 3

<details>
<summary>Clique para ver a solução</summary>

```python
import torch

def batch_encode(
    tok: ChessTokenizer,
    texts: list[str],
    max_length: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokeniza um batch de textos com padding.
    """
    batch_size = len(texts)
    input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
    
    for i, text in enumerate(texts):
        ids = tok.encode(text)
        
        # Trunca se necessário
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        # Preenche tensores
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids)
        attention_mask[i, :length] = 1  # 1 = token real
    
    return input_ids, attention_mask

# Teste
tok = ChessTokenizer()
texts = [
    "1. e4 e5",
    "1. d4 d5 2. c4",
    "1. Nf3"
]

input_ids, attention_mask = batch_encode(tok, texts, max_length=10)

print("input_ids:")
print(input_ids)
print("\nattention_mask:")
print(attention_mask)
```

Saída esperada:

```
input_ids:
tensor([[40,  1, 32, 36,  1, 32, 37,  0,  0,  0],
        [40,  1, 33, 36,  1, 33, 37,  1, 40,  2],
        [40,  1, 28, 13,  0,  0,  0,  0,  0,  0]])

attention_mask:
tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
```

</details>

---

### Solução Tarefa 4

<details>
<summary>Clique para ver a solução</summary>

```python
from collections import Counter

def analyze_token_frequency(
    tok: ChessTokenizer,
    text: str
) -> dict[int, int]:
    """
    Analisa frequência de cada token no texto.
    """
    ids = tok.encode(text)
    counter = Counter(ids)
    return dict(counter)

# Análise do dataset de pré-treino
tok = ChessTokenizer()

with open("data/pretrain.txt", "r", encoding="utf-8") as f:
    text = f.read()

freq = analyze_token_frequency(tok, text)

# Top 10 tokens mais comuns
top_10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 tokens mais comuns:")
print("Rank | Token | Contagem")
print("-" * 30)

for rank, (token_id, count) in enumerate(top_10, 1):
    token = tok.decode([token_id])
    print(f"{rank:4d} | {repr(token):6s} | {count:,}")

# Típico:
# Rank | Token | Contagem
# ------------------------------
#    1 | ' '   | ~300,000
#    2 | '1'   | ~150,000
#    3 | '.'   | ~100,000
#    4 | 'e'   | ~80,000
#    5 | '4'   | ~60,000
# etc.
```

Análise visual:

```python
import matplotlib.pyplot as plt

# Gráfico de distribuição
tokens, counts = zip(*top_10)
token_labels = [tok.decode([t]) for t in tokens]

plt.figure(figsize=(10, 5))
plt.bar(token_labels, counts)
plt.xlabel("Token")
plt.ylabel("Frequência")
plt.title("Top 10 Tokens Mais Comuns")
plt.savefig("token_frequency.png")
```

</details>

---

## Dicas

### Para Tarefa 1

```python
# Verificar se um caractere está no vocabulário
tok = ChessTokenizer()
print('x' in tok.stoi)  # False
print('e' in tok.stoi)  # True
```

### Para Tarefa 3

```python
# Padding à direita é padrão
# Use tok.pad_id (geralmente 0) para preencher
```

### Para Tarefa 4

```python
# Counter do collections é muito útil
from collections import Counter
c = Counter([1, 1, 1, 2, 2, 3])
print(c.most_common(2))  # [(1, 3), (2, 2)]
```

---

## Links Relacionados

- [[02-Modelo/tokenizer|tokenizer.py documentado]]
- [[00-Conceitos-Fundamentais/Tokenizacao|Tokenização (conceitos)]]
- [[exercicios/exercicio-02-atencao|Próximo exercício]]
