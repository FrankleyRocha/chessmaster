# ChessLM - Vault Obsidian

Bem-vindo ao vault de documentação do **ChessLM**, um projeto educacional que ensina como construir um Language Model do zero, usando xadrez como domínio de aplicação.

## O que é este Vault?

Este vault é um guia didático completo para quem quer aprender a construir um **Tiny Language Model** do zero. Através do projeto ChessLM, você vai aprender:

- Como funciona um transformer decoder-only
- Como tokenizar dados para treino
- Como estruturar um pipeline de dados
- Como treinar e fazer fine-tuning de modelos
- Como gerar texto (movimentos de xadrez) com o modelo

## Pré-requisitos

Antes de começar, você deve ter familiaridade com:

- **Python** básico/intermediário
- **PyTorch** (tensores, autograd, nn.Module)
- Conceitos básicos de **Machine Learning** (loss, gradientes, otimizadores)

## Como Navegar

### Graph View
No Obsidian, use `Ctrl+G` para visualizar o grafo de conexões entre os arquivos.

### Ordem Sugerida de Leitura

1. **[[ChessLM]]** - Visão geral do projeto
2. **[[00-Conceitos-Fundamentais/O-que-e-um-Language-Model]]** - Fundamentos
3. **[[00-Conceitos-Fundamentais/Arquitetura-Transformer]]** - Arquitetura do modelo
4. **[[00-Conceitos-Fundamentais/Tokenizacao]]** - Tokenização
5. **[[01-Data-Pipeline/Visao-Geral-Dados]]** - Pipeline de dados
6. **[[02-Modelo/Visao-Geral-Modelo]]** - Arquitetura do ChessLM
7. **[[03-Treinamento/Visao-Geral-Treinamento]]** - Treinamento
8. **[[04-Inferencia/generate]]** - Inferência
9. **[[exercicios/exercicio-01-tokenizador]]** - Prática!

## Estrutura do Vault

```
wiki/
├── 00-Conceitos-Fundamentais/    # Teoria base
├── 01-Data-Pipeline/             # Preparação de dados
├── 02-Modelo/                    # Arquitetura do modelo
├── 03-Treinamento/               # Loop de treino e fine-tuning
├── 04-Inferencia/                # Geração de movimentos
└── exercicios/                   # Prática com soluções
```

## Código Fonte

O código fonte completo está em: `../chesslm/`

Cada arquivo de documentação referencia o código correspondente.

## Filosofia do Projeto

Este projeto é inspirado no **nanoGPT** de Andrej Karpathy e segue a filosofia de:

- **Simplicidade**: Código limpo e didático
- **Educacional**: Comentários extensos, sem abstrações desnecessárias
- **Praticidade**: Funciona de verdade, gera movimentos de xadrez

## Como Usar

### Opção 1: Apenas Estudar
Leia os arquivos na ordem sugerida para entender como um Language Model é construído.

### Opção 2: Executar o Código
```bash
cd ../chesslm
pip install -r requirements.txt
# Siga as instruções em cada arquivo de documentação
```

### Opção 3: Fazer os Exercícios
Vá para [[exercicios/exercicio-01-tokenizador]] e comece a praticar!

## Links Úteis

- [nanoGPT - Karpathy](https://github.com/karpathy/nanogpt)
- [Attention is All You Need - Paper original](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
