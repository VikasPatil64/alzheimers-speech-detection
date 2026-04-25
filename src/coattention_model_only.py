"""
Co-Attention Model Definition Only (No training, no extraction)
Safe to import without triggering processing
"""

import torch
import torch.nn as nn

class CoAttentionBlock(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.15):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, y):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        cross_out, _ = self.cross_attn(x, y, y)
        x = self.norm2(x + cross_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x

class CoAttentionModel(nn.Module):
    def __init__(self, audio_dim=512, text_dim=768, proj_dim=768, num_blocks=3, num_heads=8, dropout=0.15):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.blocks = nn.ModuleList([CoAttentionBlock(proj_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim*2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 2)
        )

    def forward(self, audio_feat, text_feat):
        audio_seq = self.audio_proj(audio_feat).unsqueeze(1)
        text_seq = self.text_proj(text_feat).unsqueeze(1)
        for block in self.blocks:
            audio_seq = block(audio_seq, text_seq)
            text_seq = block(text_seq, audio_seq)
        audio_pool = audio_seq.squeeze(1)
        text_pool = text_seq.squeeze(1)
        concat = torch.cat([audio_pool, text_pool], dim=1)
        logits = self.fusion(concat)
        return logits