"""
From-scratch PyTorch Transformer for Hindi -> English translation.

This keeps the architecture you already had (it was fundamentally
correct: proper Q/K/V split, combined padding+causal mask in the
decoder, sqrt(d_model) embedding scaling) and adds the changes that
matter for limited hardware:

  - Pre-LayerNorm (optional, default on): more stable gradients, lets
    you skip a fragile LR-warmup-or-diverge regime that Post-LN needs.
  - Weight tying between the target embedding and the output
    projection: saves d_model * tgt_vocab_size parameters and is a
    well-established regularizer for small MT models.
  - PositionalEncoding now asserts on overflow instead of silently
    producing a shape-mismatched tensor, so a truncation bug fails
    loudly at the first batch instead of after an entire epoch runs.
"""

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        b, s, _ = x.size()
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x):
        b, _, s, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self._split_heads(self.W_q(Q))
        K = self._split_heads(self.W_k(K))
        V = self._split_heads(self.W_v(V))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        # Rows that are entirely masked (e.g. full-padding query position)
        # produce softmax(all -inf) = NaN; zero those out safely.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        return self.W_o(self._combine_heads(out))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the PositionalEncoding "
                f"buffer ({self.max_seq_length}). This means truncation to "
                f"max_len was NOT applied upstream -- fix the dataset "
                f"encoding step, don't raise max_seq_length blindly."
            )
        return x + self.pe[:, :seq_len]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        if self.pre_norm:
            x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        if self.pre_norm:
            xn = self.norm1(x)
            x = x + self.dropout(self.self_attn(xn, xn, xn, tgt_mask))
            xn = self.norm2(x)
            x = x + self.dropout(self.cross_attn(xn, enc_output, enc_output, src_mask))
            x = x + self.dropout(self.feed_forward(self.norm3(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
            x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask)))
            x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, d_ff, max_seq_length,
                 dropout, src_pad_idx, tgt_pad_idx, pre_norm=True,
                 tie_target_embeddings=True):
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, pre_norm)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, pre_norm)
            for _ in range(num_decoder_layers)
        ])

        # Final norm is only meaningful for Pre-LN (Post-LN already ends
        # each sublayer with a LayerNorm).
        self.pre_norm = pre_norm
        if pre_norm:
            self.enc_final_norm = nn.LayerNorm(d_model)
            self.dec_final_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        if tie_target_embeddings:
            self.fc.weight = self.decoder_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)   # [B,1,1,S]
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)  # [B,1,T,1]
        tgt_len = tgt.size(1)
        nopeak = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool))
        nopeak = nopeak.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & nopeak
        return src_mask, tgt_mask

    def encode(self, src, src_mask):
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.dropout(self.positional_encoding(x))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_final_norm(x) if self.pre_norm else x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.dropout(self.positional_encoding(x))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.dec_final_norm(x) if self.pre_norm else x

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc(dec_output)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(cfg, src_vocab_size, tgt_vocab_size, device):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        d_ff=cfg["d_ff"],
        max_seq_length=cfg["max_len"],
        dropout=cfg["dropout"],
        src_pad_idx=cfg["pad_id"],
        tgt_pad_idx=cfg["pad_id"],
        pre_norm=cfg["pre_norm"],
        tie_target_embeddings=cfg["tie_target_embeddings"],
    ).to(device)
    return model
