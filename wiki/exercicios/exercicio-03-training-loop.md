# Exercício 3: Training Loop

> Experiência prática com o loop de treinamento: experimentar hiperparâmetros e analisar resultados.

## Objetivo

Entender como diferentes hiperparâmetros afetam o treinamento e aprender a diagnosticar problemas.

---

## Tarefas

### Tarefa 1: Efeito do Learning Rate

**Objetivo:** Compare treinamentos com diferentes learning rates.

**Passos:**

1. Treine modelos com LR: `1e-5`, `3e-4`, `1e-3`
2. Monitore:
   - Velocidade de convergência
   - Loss final
   - Estabilidade (variância dos gradientes)
3. Qual LR performa melhor e por quê?

---

### Tarefa 2: Observar Overfitting

**Objetivo:** Identificar quando o modelo começa a overfittar.

**Passos:**

1. Treine por muitas iterações (ex: 20k)
2. Plote train loss vs val loss
3. Identifique o ponto de divergência
4. Implemente early stopping

---

### Tarefa 3: Implementar Logging com WandB

**Objetivo:** Adicione logging profissional ao treinamento.

```python
import wandb

def train_with_logging(cfg_model: ModelConfig, cfg_train: TrainConfig):
    """
    Loop de treino com logging WandB.
    """
    # Sua implementação
    pass
```

**Requisitos:**
- Log de loss (train/val)
- Log de learning rate
- Log de gradientes (histograma)
- Salvar modelo no WandB

---

### Tarefa 4 (Desafio): Gradient Accumulation

**Objetivo:** Implemente gradient accumulation para batch size efetivo maior.

```python
def train_with_accumulation(
    model: ChessLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int = 4
):
    """
    Treino com gradient accumulation.
    
    Acumula gradientes de N batches antes de fazer optimizer.step().
    Isso equivale a batch_size efetivo = batch_size * N.
    """
    # Sua implementação
    pass
```

---

## Soluções

### Solução Tarefa 1

<details>
<summary>Clique para ver a solução</summary>

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def compare_learning_rates():
    """Compara diferentes learning rates."""
    
    learning_rates = [1e-5, 3e-4, 1e-3]
    results = {}
    
    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"Treinando com LR = {lr}")
        print(f"{'='*50}")
        
        # Config
        cfg_model = ModelConfig()
        cfg_train = TrainConfig(
            learning_rate=lr,
            max_iters=2000,
            eval_interval=100
        )
        
        # Dados dummy para demonstração
        # (em produção, usar dados reais)
        
        # Simula curva de loss
        if lr == 1e-5:
            # LR muito baixo: convergência lenta
            losses = 4.0 * np.exp(-np.arange(2000) * 0.00001) + 1.5
        elif lr == 3e-4:
            # LR ideal: convergência suave
            losses = 4.0 * np.exp(-np.arange(2000) * 0.0005) + 0.9
        else:  # 1e-3
            # LR muito alto: instável
            base = 4.0 * np.exp(-np.arange(2000) * 0.001)
            noise = np.random.randn(2000) * 0.5
            losses = base + 1.0 + noise
        
        results[lr] = {
            "losses": losses,
            "final_loss": losses[-1],
        }
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    for lr, data in results.items():
        plt.plot(data["losses"], label=f"LR={lr:.1e}")
    
    plt.xlabel("Iteração")
    plt.ylabel("Loss")
    plt.title("Efeito do Learning Rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig("lr_comparison.png")
    plt.show()
    
    # Resumo
    print("\n" + "="*60)
    print("Resumo:")
    print("="*60)
    print(f"{'LR':<12} | {'Loss Final':<12} | {'Observação'}")
    print("-"*60)
    print(f"1e-5        | {results[1e-5]['final_loss']:.4f}       | Muito lento")
    print(f"3e-4        | {results[3e-4]['final_loss']:.4f}       | Ideal (conv suave)")
    print(f"1e-3        | {results[1e-3]['final_loss']:.4f}       | Instável")
    
    return results

# Executar
results = compare_learning_rates()
```

**Conclusões:**

| LR | Efeito | Loss Final |
|----|--------|------------|
| 1e-5 | Muito conservador, convergência lenta | ~2.5 |
| 3e-4 | Ideal, convergência suave | ~0.9 |
| 1e-3 | Alto demais, training instável | ~1.5 (com ruído) |

</details>

---

### Solução Tarefa 2

<details>
<summary>Clique para ver a solução</summary>

```python
import matplotlib.pyplot as plt

def detect_overfitting():
    """Treina até detectar overfitting."""
    
    cfg = TrainConfig(max_iters=20000, eval_interval=100)
    
    # Simulação de curvas de loss
    iters = np.arange(0, 20001, 100)
    
    # Train loss: continua diminuindo
    train_loss = 4.0 * np.exp(-iters * 0.0002) + 0.3
    
    # Val loss: diminui, depois aumenta (overfitting!)
    val_loss_base = 4.0 * np.exp(-iters * 0.00015) + 0.5
    # Adiciona "U-shape" para simular overfitting
    overfit_start = 10000
    overfit_curve = 0.002 * np.maximum(0, iters - overfit_start)**1.5 / 1000
    val_loss = val_loss_base + overfit_curve
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(iters, train_loss, label="Train Loss", linewidth=2)
    plt.plot(iters, val_loss, label="Val Loss", linewidth=2)
    
    # Marca ponto de overfitting
    best_idx = np.argmin(val_loss)
    plt.axvline(x=iters[best_idx], color='red', linestyle='--', 
                label=f'Best val @ iter {iters[best_idx]}')
    
    plt.xlabel("Iteração")
    plt.ylabel("Loss")
    plt.title("Detecção de Overfitting")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("overfitting.png")
    plt.show()
    
    return iters[best_idx]

# Early stopping implementado
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        """
        Returns True se deve parar.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Uso no training loop
early_stop = EarlyStopping(patience=10)

for it in range(max_iters):
    # ... treino ...
    
    if it % eval_interval == 0:
        val_loss = evaluate(model, val_loader)
        
        if early_stop(val_loss):
            print(f"Early stopping at iteration {it}")
            print(f"Best val loss: {early_stop.best_loss:.4f}")
            break
```

**Análise:**

```
Iteração | Train Loss | Val Loss | Gap  | Status
---------|------------|----------|------|--------
0        | 4.00       | 4.00     | 0.00 | OK
5000     | 1.50       | 1.65     | 0.15 | OK
10000    | 0.80       | 0.95     | 0.15 | OK
12000    | 0.60       | 0.85     | 0.25 | ⚠️ Overfitting
15000    | 0.45       | 1.10     | 0.65 | ❌ Parar!
```

</details>

---

### Solução Tarefa 3

<details>
<summary>Clique para ver a solução</summary>

```python
import wandb
from typing import Optional

def train_with_logging(
    cfg_model: ModelConfig,
    cfg_train: TrainConfig,
    project: str = "chesslm",
    name: Optional[str] = None
):
    """
    Loop de treino com logging WandB.
    """
    # Inicializa WandB
    wandb.init(
        project=project,
        name=name,
        config={
            "model": cfg_model.__dict__,
            "train": cfg_train.__dict__,
        }
    )
    
    # Setup
    device = cfg_train.device
    model = ChessLM(cfg_model).to(device)
    optimizer = model.configure_optimizers(cfg_train)
    
    # DataLoader (assumindo que existe)
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    
    # Watch model para histogramas de gradientes
    wandb.watch(model, log_freq=100)
    
    best_val_loss = float("inf")
    
    for it in range(cfg_train.max_iters):
        lr = get_lr(it, cfg_train)
        
        # Train step
        x, y = train_loader.get_batch()
        _, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Log gradientes
        if it % 100 == 0:
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            
            wandb.log({
                "gradients/mean_norm": np.mean(grad_norms),
                "gradients/max_norm": np.max(grad_norms),
            })
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
        optimizer.step()
        
        # Log de treino
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": lr,
            "train/iter": it,
        })
        
        # Validação
        if it % cfg_train.eval_interval == 0:
            val_loss = evaluate(model, val_loader)
            
            wandb.log({
                "val/loss": val_loss,
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Salva modelo no WandB
                torch.save(model.state_dict(), "best_model.pt")
                wandb.save("best_model.pt")
                
                wandb.log({"val/best_loss": best_val_loss})
    
    # Finaliza
    wandb.finish()

def evaluate(model, loader, n_batches=100):
    """Avalia modelo no validation set."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = loader.get_batch()
            _, loss = model(x, y)
            losses.append(loss.item())
    
    model.train()
    return np.mean(losses)

# Dashboard WandB mostrará:
# - Gráfico de train/val loss
# - Learning rate ao longo do tempo
# - Histogramas de gradientes
# - Comparação entre runs
```

**Configuração WandB:**

```bash
# Instalar
pip install wandb

# Login
wandb login

# Executar treino
python train.py

# Ver resultados
# https://wandb.ai/seu-usuario/chesslm
```

</details>

---

### Solução Tarefa 4

<details>
<summary>Clique para ver a solução</summary>

```python
def train_with_accumulation(
    model: ChessLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    accumulation_steps: int = 4,
    scaler: torch.cuda.GradScaler | None = None
):
    """
    Treino com gradient accumulation.
    
    Exemplo:
        batch_size = 16
        accumulation_steps = 4
        → batch efetivo = 64
    """
    model.train()
    
    # Batch size efetivo
    effective_batch_size = cfg.batch_size * accumulation_steps
    print(f"Batch size: {cfg.batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    optimizer.zero_grad()
    
    for it in range(cfg.max_iters):
        # Loop de acumulação
        for acc_step in range(accumulation_steps):
            x, y = loader.get_batch()
            
            # Forward
            with torch.amp.autocast(device_type=cfg.device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            
            # Escala loss para que gradientes acumulem corretamente
            loss = loss / accumulation_steps
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Agora temos gradientes de accumulation_steps batches acumulados
        
        # Clip gradientes
        if scaler:
            scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        # Optimizer step (uma vez!)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Zera gradientes para próximo ciclo
        optimizer.zero_grad()
        
        # Update LR
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        
        # Log
        if it % cfg.log_interval == 0:
            print(f"iter {it} | loss {loss.item() * accumulation_steps:.4f} | lr {lr:.2e}")

# Versão alternativa: usar Fear of Missing Gradients
def train_with_accumulation_v2(
    model,
    loader,
    optimizer,
    cfg,
    accumulation_steps=4
):
    """
    Versão com loss tracking mais preciso.
    """
    losses = []
    optimizer.zero_grad()
    
    for it in range(cfg.max_iters):
        epoch_loss = 0.0
        
        for acc_step in range(accumulation_steps):
            x, y = loader.get_batch()
            _, loss = model(x, y)
            
            epoch_loss += loss.item()
            loss = loss / accumulation_steps
            loss.backward()
        
        # Média real da loss
        avg_loss = epoch_loss / accumulation_steps
        losses.append(avg_loss)
        
        # Clip e step
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    
    return losses

# Comparação de memória:
# Sem accumulation:
#   VRAM = batch_size × seq_len × model_size
#   Ex: 64 × 512 × 5M = muita VRAM
#
# Com accumulation (accumulation_steps=4):
#   VRAM = (batch_size/4) × seq_len × model_size
#   Mas batch efetivo continua 64!
```

</details>

---

## Dicas

### Para Tarefa 1

```python
# Quando LR é muito alto, gradientes podem explodir
# Verificar: grad_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
```

### Para Tarefa 2

```python
# Overfitting acontece quando:
# - Train loss continua diminuindo
# - Val loss para de diminuir ou aumenta
# - Gap train/val aumenta
```

### Para Tarefa 3

```python
# WandB gratis para projetos pessoais
# Organize runs com tags: wandb.init(tags=["experiment1"])
```

### Para Tarefa 4

```python
# Cuidado com BatchNorm durante accumulation
# May need to set model.train() e desativar BatchNorm updates
```

---

## Links Relacionados

- [[03-Treinamento/train|train.py documentado]]
- [[03-Treinamento/Visao-Geral-Treinamento|Visão Geral]]
- [[exercicios/exercicio-04-melhorias|Próximo exercício]]
