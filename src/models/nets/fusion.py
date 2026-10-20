"""Dynamic spatiotemporal fusion; DSTM is the legacy DSFM class name."""

import torch
from torch import nn


class DSTM(nn.Module):
    def __init__(self, dim, sr_ratio=4, mamba_interval=3):
        super().__init__()
        self.scale = dim**-0.5
        self.sr_ratio = sr_ratio

        def get_sr(dim, sr_ratio):
            if sr_ratio > 1:
                sr = nn.Sequential(
                    nn.Conv1d(dim, dim // sr_ratio, kernel_size=1),
                    nn.BatchNorm1d(dim // sr_ratio),
                    nn.GELU(),
                )
            else:
                sr = nn.Identity()
            return sr

        self.v = nn.Sequential(
            get_sr(dim, sr_ratio),
            nn.Conv1d(dim // sr_ratio, dim // sr_ratio, kernel_size=1),
            nn.BatchNorm1d(dim // sr_ratio),
        )

        self.k = nn.Sequential(
            get_sr(dim, sr_ratio),
            nn.Conv1d(dim // sr_ratio, dim // sr_ratio, kernel_size=1),
            nn.BatchNorm1d(dim // sr_ratio),
        )

        self.temporal_proj = nn.Conv2d((mamba_interval - 1), 2, 1)
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.fusion = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
        )

        # dropout
        self.residual_dropout = nn.Dropout(0.1)

    def forward(self, current_feature, temporal_features):
        B, N, T, C = temporal_features.shape

        v = temporal_features[:, -1]  # [B, T, C]
        other_features = temporal_features[:, :-1]  # [B, 2, T, C]

        qk_features = self.temporal_proj(other_features)  # [B, 2, T, C]
        qk_features = self.norm(qk_features)

        q, k = qk_features.chunk(2, dim=1)
        q = q.squeeze(1)  # [B, T, C]
        k = k.squeeze(1)  # [B, T, C]

        v = v.transpose(1, 2)  # [B, C, T]
        k = k.transpose(1, 2)  # [B, C, T]

        k = self.k(k)  # [B, C/sr_ratio, T]
        v = self.v(v)  # [B, C/sr_ratio, T]

        k = k.transpose(1, 2)  # [B, T, C/sr_ratio]

        attn = torch.matmul(k, v) * self.scale  # [B, T, T]
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, q)  # [B, T, C]

        fusion_input = torch.cat([current_feature, x], dim=-1)
        fusion_input = fusion_input.transpose(1, 2)
        out = self.fusion(fusion_input)
        out = out.transpose(1, 2)

        out = self.residual_dropout(out) + current_feature
        out = self.norm2(out)

        return out
