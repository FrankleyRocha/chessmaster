# Solução: Exercício 3 - Training Loop

Soluções completas para o [[exercicios/exercicio-03-training-loop|Exercício 3]].

## Tarefa 1: Efeito do Learning Rate

```python
#!/usr/bin/env python
"""Compara diferentes learning rates."""

import numpy as np
import matplotlib.pyplot as plt

def simulate_training(lr: float, n_iters: int = 2000) -> np.ndarray:
    """Simula curva de loss para um learning rate."""
    
    # Parâmetros da simulação
    initial_loss = 4.0
    final_loss = 0.9
    noise_scale = 0.1
    
    # Taxa de convergência depende do LR
    if lr < 1e-4:
        # LR muito baixo: convergência lenta
        convergence_rate = 0.00005
    elif lr > 5e-4:
        # LR muito alto: instável
        convergence_rate = 0.002
        noise_scale = 0.3  # Mais ruído
    else:
        # LR ideal
        convergence_rate = 0.0008
        noise_scale = 0.05
    
    # Simula loss
    t = np.arange(n_iters)
    base_loss = initial_loss * np.exp(-t * convergence_rate) + final_loss
    noise = np.random.randn(n_iters) * noise_scale
    loss = base_loss + noise
    
    # Limita valores
    loss = np.maximum(loss, 0.1)
    
    return loss

def plot_lr_comparison():
    """Plota comparação de LRs."""
    
    learning_rates = [1e-5, 3e-4, 1e-3]
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(12, 6))
    
    for lr, color in zip(learning_rates, colors):
        loss = simulate_training(lr)
        plt.plot(loss, color=color, alpha=0.7, linewidth=1.5, label=f"LR={lr:.1e}")
    
    plt.xlabel("Iteração")
    plt.ylabel("Loss")
    plt.title("Efeito do Learning Rate na Convergência")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    
    plt.tight_layout()
    plt.savefig("lr_comparison.png", dpi=150)
    plt.show()
    
    # Resumo
    print("\nResumo:")
    print("="*50)
    print(f"{'LR':<12} | {'Loss Final':<12} | {'Observação'}")
    print("-"*50)
    
    for lr in learning_rates:
        final_loss = simulate_training(lr)[-100:].mean()
        
        if lr < 1e-4:
            obs = "Muito lento"
        elif lr > 5e-4:
            obs = "Instável"
        else:
            obs = "Ideal"
        
        print(f"{lr:.1e}    | {final_loss:.4f}       | {obs}")

if __name__ == "__main__":
    np.random.seed(42)
    plot_lr_comparison()
```

---

## Tarefa 2: Observar Overfitting

```python
#!/usr/bin/env python
"""Detecta e visualiza overfitting."""

import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping:
    """Implementação de early stopping."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_iter = 0
    
    def __call__(self, val_loss: float, iteration: int) -> bool:
        """Verifica se deve parar."""
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_iter = iteration
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False

def simulate_overfitting(n_iters: int = 20000):
    """Simula curvas de loss com overfitting."""
    
    iters = np.arange(0, n_iters + 1, 100)
    
    # Train loss: continua diminuindo
    train_loss = 4.0 * np.exp(-iters * 0.0002) + 0.3
    
    # Val loss: diminui, depois aumenta
    val_base = 4.0 * np.exp(-iters * 0.00015) + 0.5
    overfit_start = 10000
    overfit_factor = 0.003 * np.maximum(0, iters - overfit_start)**1.3 / 1000
    val_loss = val_base + overfit_factor
    
    return iters, train_loss, val_loss

def demonstrate_early_stopping():
    """Demonstra early stopping."""
    
    iters, train_loss, val_loss = simulate_overfitting()
    
    # Early stopping
    early_stop = EarlyStopping(patience=10, min_delta=0.005)
    stop_iter = None
    
    for i, (it, vl) in enumerate(zip(iters, val_loss)):
        if early_stop(vl, it):
            stop_iter = it
            break
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(iters, train_loss, label="Train Loss", linewidth=2, color='blue')
    plt.plot(iters, val_loss, label="Val Loss", linewidth=2, color='orange')
    
    # Marca melhor ponto
    plt.axvline(x=early_stop.best_iter, color='green', linestyle='--', 
                label=f'Best Model (iter {early_stop.best_iter})', alpha=0.7)
    
    # Marca parada
    if stop_iter:
        plt.axvline(x=stop_iter, color='red', linestyle=':', 
                    label=f'Early Stop (iter {stop_iter})', alpha=0.7)
    
    # Região de overfitting
    plt.axvspan(early_stop.best_iter, iters[-1], alpha=0.2, color='red', 
                label='Overfitting Region')
    
    plt.xlabel("Iteração")
    plt.ylabel("Loss")
    plt.title("Detecção de Overfitting com Early Stopping")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("overfitting.png", dpi=150)
    plt.show()
    
    print("\nAnálise:")
    print("="*50)
    print(f"Melhor iter: {early_stop.best_iter}")
    print(f"Melhor val loss: {early_stop.best_loss:.4f}")
    print(f"Parou em: {stop_iter}")
    print(f"Iterações poupadas: {iters[-1] - stop_iter}")

if __name__ == "__main__":
    demonstrate_early_stopping()
```

---

## Tarefa 3: Logging com WandB

```python
#!/usr/bin/env python
"""Treino com logging WandB."""

import torch
import numpy as np
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB não instalado. pip install wandb")

def train_with_wandb(
    project: str = "chesslm",
    name: str = "experiment",
    config: dict = None,
    dry_run: bool = False
):
    """
    Loop de treino com logging WandB.
    
    Args:
        project: Nome do projeto WandB
        name: Nome da run
        config: Configurações para logar
        dry_run: Se True, simula sem WandB real
    """
    
    if not WANDB_AVAILABLE or dry_run:
        print("Modo dry run (sem WandB real)")
        
        # Simula logs
        for it in range(0, 1001, 50):
            train_loss = 4.0 * np.exp(-it * 0.001) + np.random.randn() * 0.1
            val_loss = train_loss + np.random.randn() * 0.2
            lr = 3e-4 * max(0.1, 1 - it / 500)
            
            print(f"iter {it:4d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e}")
        
        return
    
    # WandB real
    wandb.init(
        project=project,
        name=name,
        config=config or {},
    )
    
    # Log de exemplo
    for it in range(1001):
        # Simula métricas
        train_loss = 4.0 * np.exp(-it * 0.001) + np.random.randn() * 0.1
        val_loss = train_loss + np.random.randn() * 0.2
        lr = 3e-4 * max(0.1, 1 - it / 500)
        
        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/lr": lr,
            "train/iter": it,
        })
    
    wandb.finish()

# Exemplo de uso real
def real_training_example():
    """Exemplo de uso real com ChessLM."""
    
    config = {
        "model": {
            "n_embd": 256,
            "n_layer": 6,
            "n_head": 8,
            "block_size": 512,
        },
        "train": {
            "batch_size": 64,
            "learning_rate": 3e-4,
            "max_iters": 50000,
        }
    }
    
    print("Para usar WandB:")
    print("1. pip install wandb")
    print("2. wandb login")
    print("3. Execute este script")
    print("\nComando:")
    print("  python train_with_wandb.py --project chesslm --name my_run")
    
    # Dry run para demonstração
    train_with_wandb(dry_run=True)

if __name__ == "__main__":
    real_training_example()
```

---

## Tarefa 4: Gradient Accumulation

```python
#!/usr/bin/env python
"""Implementação de gradient accumulation."""

import torch
import torch.nn as nn
import time

def train_without_accumulation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    batch_size: int,
    n_iters: int
):
    """Treino padrão."""
    
    model.train()
    losses = []
    
    for _ in range(n_iters):
        # Batch
        idx = torch.randint(0, len(data) - 128, (batch_size,))
        x = data[idx]
        y = data[idx + 1]
        
        # Forward
        _, loss = model(x, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses

def train_with_accumulation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    batch_size: int,
    accumulation_steps: int,
    n_iters: int
):
    """
    Treino com gradient accumulation.
    
    Batch efetivo = batch_size * accumulation_steps
    """
    
    model.train()
    losses = []
    
    small_batch = batch_size // accumulation_steps
    
    for _ in range(n_iters):
        epoch_loss = 0.0
        
        # Acumula gradientes
        for _ in range(accumulation_steps):
            # Batch menor
            idx = torch.randint(0, len(data) - 128, (small_batch,))
            x = data[idx]
            y = data[idx + 1]
            
            # Forward
            _, loss = model(x, y)
            loss = loss / accumulation_steps  # Escala
            
            # Backward (acumula)
            loss.backward()
            
            epoch_loss += loss.item() * accumulation_steps
        
        # Step único
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(epoch_loss / accumulation_steps)
    
    return losses

def benchmark_accumulation():
    """Compara treino com e sem accumulation."""
    
    print("Benchmark: Gradient Accumulation")
    print("="*60)
    
    from model.config import ModelConfig
    from model.model import ChessLM
    
    # Setup
    cfg = ModelConfig(n_embd=128, n_layer=2)  # Menor para teste
    model = ChessLM(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dados dummy
    data = torch.randint(0, 64, (10000,))
    
    batch_size = 64
    accumulation_steps = 4
    n_iters = 100
    
    print(f"\nBatch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    print(f"Iterations: {n_iters}")
    
    # Sem accumulation
    print("\n[1] Treino padrão...")
    model_std = ChessLM(cfg)
    opt_std = torch.optim.Adam(model_std.parameters(), lr=1e-3)
    
    start = time.time()
    losses_std = train_without_accumulation(model_std, opt_std, data, batch_size, n_iters)
    time_std = time.time() - start
    
    # Com accumulation
    print("[2] Treino com accumulation...")
    model_acc = ChessLM(cfg)
    opt_acc = torch.optim.Adam(model_acc.parameters(), lr=1e-3)
    
    start = time.time()
    losses_acc = train_with_accumulation(model_acc, opt_acc, data, batch_size, accumulation_steps, n_iters)
    time_acc = time.time() - start
    
    # Resultados
    print("\n" + "="*60)
    print("Resultados:")
    print("="*60)
    print(f"{'Método':<25} | {'Tempo':<10} | {'Loss Final':<12}")
    print("-"*60)
    print(f"{'Padrão':<25} | {time_std:>8.2f}s | {losses_std[-1]:>10.4f}")
    print(f"{'Accumulation (4x)':<25} | {time_acc:>8.2f}s | {losses_acc[-1]:>10.4f}")
    
    print("\nVantagens do Accumulation:")
    print("  - Batch efetivo maior com menos VRAM")
    print("  - Treinar modelos grandes em hardware limitado")
    print("  - Ligeiramente mais lento por iteração")

if __name__ == "__main__":
    benchmark_accumulation()
```

Execute:

```bash
python exercises/solution_03.py
```
