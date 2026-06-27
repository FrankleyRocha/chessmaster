# ChessLM 🏆

Language model treinado para jogar xadrez no estilo dos maiores jogadores da história.

## Arquitetura

- **Modelo**: GPT-like, transformer decoder-only
- **Tokenização**: por caractere (~50 tokens de vocabulário)
- **Input**: partida em notação PGN até o momento
- **Output**: próximo caractere (geração autoregressiva)

## Estratégia de treino

1. **Pré-treino**: Lichess Database filtrado (Elo ≥ 2500, apenas vitórias)
2. **Fine-tuning**: Partidas de Fischer, Kasparov, Tal, Capablanca, Pragg e Magnus

## Estrutura do projeto

```
chesslm/
├── data/
│   ├── download_lichess.py      # Download e filtragem do dataset Lichess
│   ├── download_players.py      # Download das partidas dos jogadores (PGN Mentor)
│   └── prepare_dataset.py       # Preparação e serialização do dataset
├── model/
│   ├── tokenizer.py             # Tokenizador por caractere
│   ├── model.py                 # Arquitetura GPT-like
│   └── config.py                # Hiperparâmetros
├── training/
│   ├── train.py                 # Loop de treino principal
│   └── finetune.py              # Fine-tuning nos jogadores
├── inference/
│   └── generate.py              # Geração de movimentos
├── utils/
│   └── pgn_utils.py             # Utilitários para parsing de PGN
├── requirements.txt
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso rápido

### 1. Baixar dados

```bash
# Baixar e filtrar dataset Lichess (Elo 2500+, vitórias)
python data/download_lichess.py --output data/pretrain.txt --month 2024-01 --max-games 10000

# Baixar partidas dos jogadores históricos
python data/download_players.py --output data/players.txt
```

### 2. Preparar datasets

```bash
# Preparar dataset de pré-treino (gera data/pretrain_train.npy e data/pretrain_val.npy)
python data/prepare_dataset.py --input data/pretrain.txt --name pretrain

# Preparar dataset de fine-tuning (gera data/finetune_train.npy e data/finetune_val.npy)
python data/prepare_dataset.py --input data/players.txt --name finetune
```

### 3. Treinar

```bash
# Pré-treino (lê data/pretrain_train.npy e data/pretrain_val.npy)
python training/train.py

# Fine-tuning (lê data/finetune_train.npy e data/finetune_val.npy)
python training/finetune.py --checkpoint checkpoints/pretrain_final.pt
```

**Importante:** Os scripts de treino esperam que os arquivos `.npy` tenham sido gerados pelo `prepare_dataset.py`. Se os arquivos não existirem, uma mensagem de erro indicará como proceder.

### 3. Gerar movimentos

```bash
python inference/generate.py --prompt "1. e4 e5 2. Nf3" --moves 5
```

## Hiperparâmetros padrão

| Parâmetro   | Valor |
|-------------|-------|
| vocab_size  | 64    |
| block_size  | 512   |
| n_embd      | 256   |
| n_head      | 8     |
| n_layer     | 6     |
| dropout     | 0.1   |
| batch_size  | 64    |
| lr          | 3e-4  |

## Hardware recomendado

- **Pré-treino**: GPU com ≥ 8GB VRAM (RTX 3060, RTX 4060, etc.)
- **Fine-tuning**: mesma GPU, dataset menor — bem mais rápido
- **Inferência**: roda em CPU sem problemas
