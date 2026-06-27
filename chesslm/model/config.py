"""
Configurações do ChessLM.
Dois perfis: 'pretrain' (dataset grande) e 'finetune' (jogadores históricos).
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Arquitetura
    vocab_size: int   = 512     # definido pelo tokenizador BPE
    block_size: int   = 512     # contexto máximo em tokens (~1500 chars com BPE)
    n_embd:     int   = 256     # dimensão do embedding
    n_head:     int   = 8       # cabeças de atenção (n_embd % n_head == 0)
    n_layer:    int   = 6       # blocos transformer
    dropout:    float = 0.1     # dropout em atenção e projeções

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) deve ser divisível por n_head ({self.n_head})"


@dataclass
class TrainConfig:
    # Dados
    dataset_name:    str   = "pretrain"
    data_dir:        str   = "data"
    val_split:       float = 0.05       # 5% para validação
    tokenizer_path:  str   = "tokenizer_bpe.json"

    # Treino
    batch_size:      int   = 64
    max_iters:       int   = 50_000
    eval_interval:   int   = 500
    eval_iters:      int   = 100
    log_interval:    int   = 100

    # Otimizador
    learning_rate:   float = 3e-4
    weight_decay:    float = 0.1
    beta1:           float = 0.9
    beta2:           float = 0.95
    grad_clip:       float = 1.0

    # Learning rate schedule (cosine decay com warmup)
    warmup_iters:    int   = 1000
    lr_decay_iters:  int   = 50_000
    min_lr:          float = 3e-5

    # Checkpoints
    checkpoint_dir:  str   = "checkpoints"
    checkpoint_name: str   = "pretrain"
    save_interval:   int   = 2000

    # Device
    device:          str   = "cuda"     # "cuda" ou "cpu"
    dtype:           str   = "bfloat16" # "float32", "float16", "bfloat16"
    compile:         bool  = True       # torch.compile (PyTorch 2.0+)


@dataclass
class FinetuneConfig(TrainConfig):
    # Override para fine-tuning
    dataset_name:    str   = "finetune"
    checkpoint_name: str   = "finetune"

    # Learning rate bem menor para não destruir o pré-treino
    learning_rate:   float = 3e-5
    min_lr:          float = 3e-6
    warmup_iters:    int   = 100
    max_iters:       int   = 5_000
    lr_decay_iters:  int   = 5_000
    save_interval:   int   = 500

    # Checkpoint de pré-treino para carregar
    pretrain_checkpoint: str = "checkpoints/pretrain_final.pt"


# Instâncias prontas para importar
DEFAULT_MODEL  = ModelConfig()
PRETRAIN_CFG   = TrainConfig()
FINETUNE_CFG   = FinetuneConfig()
