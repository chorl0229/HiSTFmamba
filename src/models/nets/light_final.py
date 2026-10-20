import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.nets.mamba_block import VisionMamba
from src.models.nets.histf_mamba import HiSTF_Mamba


class LightT2M(nn.Module):
    def __init__(
        self,
        num_layers=6,
        latent_dim=512,
        dropout=0.1,
        clip_dim=512,
        cond_mask_prob=0.1,
        input_dim=263,
        mamba_interval=3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.clip_dim = clip_dim
        self.cond_mask_prob = cond_mask_prob
        self.input_dim = input_dim

        if latent_dim != 512 or input_dim not in (251, 263):
            raise ValueError(
                "HiSTF requires latent_dim=512 and input_dim=263 (HumanML3D) or 251 (KIT-ML)."
            )
        self.max_motion_len = 196
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.input_process = InputProcess(self.input_dim, self.latent_dim)

        self.seqTransEncoder = HiSTF_Mamba(
            depth=self.num_layers,
            embed_dim=self.latent_dim,
            d_state=16,
            ssm_cfg=None,
            drop_path_rate=0.1,
            norm_epsilon=1e-5,
            rms_norm=True,
            initializer_cfg=None,
            fused_add_norm=True,
            residual_in_fp32=True,
            device=None,
            dtype=None,
            if_bidirectional=False,
            final_pool_type="none",
            if_rope=False,
            if_rope_residual=False,
            flip_img_sequences_ratio=-1.0,
            if_bimamba=False,
            bimamba_type="v2",
            if_divide_out=True,
            init_layer_scale=None,
            mamba_interval=mamba_interval,
            dataset="humanml" if input_dim == 263 else "kit",
        )

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

    def forward(self, motion, motion_mask, timestep, text):
        length = motion.shape[1]
        if not 1 <= length <= self.max_motion_len:
            raise ValueError("HiSTF supports 1 to 196 motion frames.")
        # Spatial scans use time as the channel axis and require 197 tokens.
        motion = F.pad(motion, (0, 0, 0, self.max_motion_len - length))
        motion = self.input_process(motion)
        emb = self.embed_timestep(timestep)
        text = self.embed_text(text["text_emb"]).unsqueeze(1).clone()

        emb = emb + text

        xseq = torch.cat((emb, motion), axis=1)  # [ bs, seqlen+1, d]
        xseq = xseq.permute(1, 0, 2).contiguous()
        xseq = self.sequence_pos_encoder(xseq)
        xseq = xseq.permute(1, 0, 2).contiguous()
        output = self.seqTransEncoder.forward(xseq)
        output = output[:, 1:, :]

        out = self.output_process(output)

        return out[:, :length]

    def mask_cond(self, bs, force_mask=False):
        """
        mask motion condition , return contitional motion index in the batch
        """
        if force_mask:
            cond_indices = torch.empty(0)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(
                    bs,
                )
                * self.cond_mask_prob
            )  # 1-> use null_cond, 0-> use real cond
            mask = 1.0 - mask
            cond_indices = torch.nonzero(mask).squeeze(-1)
        else:
            cond_indices = torch.arange(bs)

        return cond_indices


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe.clone().detach())

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps].clone())


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()

        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        output = self.poseFinal(output)
        return output
