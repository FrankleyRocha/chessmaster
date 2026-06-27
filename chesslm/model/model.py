"""
ChessLM — Arquitetura GPT-like (decoder-only transformer).

Baseado em nanoGPT (Karpathy), adaptado para notação PGN por caractere.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import ModelConfig


# ─────────────────────────────────────────────
#  Blocos básicos
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention com máscara causal."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.dropout = cfg.dropout

        # Projeções Q, K, V num único linear para eficiência
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Máscara causal (triangular inferior)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequência, embedding

        # Calcula Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head

        # Reorganiza para (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Atenção escalonada
        scale = 1.0 / math.sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) * scale

        # Máscara causal
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Agrega valores
        y = att @ v                                          # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, C)

        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward com ativação GELU."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2  = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """Bloco transformer: LayerNorm → Atenção → LayerNorm → MLP (Pre-LN)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ─────────────────────────────────────────────
#  Modelo principal
# ─────────────────────────────────────────────

class ChessLM(nn.Module):
    """
    Language model para xadrez.

    Recebe sequência de tokens (caracteres PGN) e prevê o próximo token.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),     # token embeddings
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),     # position embeddings
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: embedding e lm_head compartilham pesos
        self.transformer.wte.weight = self.lm_head.weight

        # Inicialização
        self.apply(self._init_weights)
        # Escala residual conforme GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ChessLM inicializado — {n_params/1e6:.1f}M parâmetros")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) tensor de IDs de tokens
            targets: (B, T) tensor de IDs alvo (para cálculo de loss)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar (se targets fornecido) ou None
        """
        B, T = idx.size()
        assert T <= self.cfg.block_size, \
            f"Sequência de tamanho {T} excede block_size={self.cfg.block_size}"

        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Embeddings de token + posição
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Blocos transformer
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Gera novos tokens autoregressivamente.

        Args:
            idx:            (B, T) contexto inicial
            max_new_tokens: quantos tokens gerar
            temperature:    > 1 = mais aleatório, < 1 = mais determinístico
            top_k:          se definido, filtra para os k tokens mais prováveis

        Returns:
            (B, T + max_new_tokens) sequência estendida
        """
        for _ in range(max_new_tokens):
            # Trunca contexto se necessário
            idx_cond = idx if idx.size(1) <= self.cfg.block_size \
                       else idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # último token

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    def configure_optimizers(self, cfg) -> torch.optim.Optimizer:
        """
        AdamW com weight decay apenas em parâmetros 2D (pesos),
        não em biases e LayerNorms.
        """
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Remove parâmetros que não existem (caso de weight tying)
        decay = decay & param_dict.keys()
        no_decay = no_decay & param_dict.keys()
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)],
             "weight_decay": cfg.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
        )
        return optimizer
