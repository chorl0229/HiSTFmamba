# Portions derived from Vision Mamba and timm.
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# See THIRD_PARTY_NOTICES.md for retained upstream license terms.
from .fusion import DSTM
from .spatial import Part_Mamba_Conv_Stage

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights


from mamba_ssm.modules.mamba_simple import Mamba

import math

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from src.models.nets.whole_part import parts2whole


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), (
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class Block_No_Res(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()

        self.mixer = mixer_cls(dim)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_state,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def create_no_res_block(
    d_model,
    d_state,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block_No_Res(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class Daul_Spatial_Mamba(nn.Module):
    def __init__(
        self,
        d_state,
        layer_idx,
        bimamba_type,
        if_divide_out,
        init_layer_scale,
        residual_in_fp32,
        ssm_cfg,
        factory_kwargs,
        norm_cls=nn.LayerNorm,
        drop_path=0.0,
        dataset="humanml",
        mamba_interval=3,
    ):
        super(Daul_Spatial_Mamba, self).__init__()

        if dataset.lower() == "humanml":
            input_dim = 263
            dims_dict = {
                "Root": 7,
                "R_Leg": 50,
                "L_Leg": 50,
                "Backbone": 60,
                "R_Arm": 60,
                "L_Arm": 60,
            }
        else:  # kit
            input_dim = 251
            dims_dict = {
                "Root": 7,
                "R_Leg": 62,
                "L_Leg": 62,
                "Backbone": 48,
                "R_Arm": 48,
                "L_Arm": 48,
            }

        self.input_projection = nn.Linear(512, input_dim)
        self.norm = RMSNorm(512, eps=1e-5, **factory_kwargs)

        self.conv_stage = Part_Mamba_Conv_Stage(
            embed_dim=128, activation="relu", K=4, dataset=dataset
        )

        self.mixer_part = create_no_res_block(
            d_model=197,
            d_state=d_state,
            ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5,
            drop_path=0.1,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            layer_idx=layer_idx,
            device=factory_kwargs.get("device", None),
            dtype=factory_kwargs.get("dtype", None),
            if_bimamba=if_divide_out,
            bimamba_type="v3",
            if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale,
        )

        self.mixer_whole = create_no_res_block(
            d_model=197,
            d_state=d_state,
            ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5,
            drop_path=0.1,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            layer_idx=layer_idx,
            device=factory_kwargs.get("device", None),
            dtype=factory_kwargs.get("dtype", None),
            if_bimamba=if_divide_out,
            bimamba_type="v4",
            if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale,
        )

        self.norm = norm_cls(512)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.combine_projection = nn.Linear(input_dim, 512)

        self.parts = ["Root", "R_Leg", "L_Leg", "Backbone", "R_Arm", "L_Arm"]
        self.num_parts = len(self.parts)

        self.projections_out = nn.ModuleList(
            [nn.Linear(128, dims_dict[part]) for part in self.parts]
        )

        self.residual_in_fp32 = residual_in_fp32
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.dataset = dataset
        self.mamba_interval = mamba_interval
        self.dstf = DSTM(dim=512, mamba_interval=self.mamba_interval)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        temporal_features: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """

        assert hidden_states is not None, "Input hidden_states cannot be None"

        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        batch_size = hidden_states.shape[0]
        emb = hidden_states[:, 0:1, :]

        hidden_states_part = self.conv_stage(
            hidden_states
        )  # (batch_size * 6, 128, sequence_length)

        hidden_states_part = self.mixer_part(
            hidden_states_part, inference_params=inference_params
        )  # (batch_size * 6, 128, sequence_length)

        hidden_states = hidden_states.permute(0, 2, 1).contiguous()  # (64, 512, 197)

        hidden_states_whole = self.mixer_whole(
            hidden_states
        )  # (batch_size, sequence_length -1, 263)

        hidden_states_part = hidden_states_part.permute(
            0, 2, 1
        ).contiguous()  # (batch_size*6, 197, 128)
        hidden_states_part = hidden_states_part[:, 1:, :]  # (batch_size*6, 196, 128)

        hidden_states_part = hidden_states_part.view(
            batch_size, self.num_parts, 196, 128
        )  # (64, 6, 196, 128)

        parts_proj_mamba = [hidden_states_part[:, p, :, :] for p in range(self.num_parts)]

        parts_out = [proj(part) for proj, part in zip(self.projections_out, parts_proj_mamba)]

        hidden_states_part = parts2whole(
            parts_out, mode="t2m" if self.dataset == "humanml" else "kit"
        )

        hidden_states_part = self.combine_projection(hidden_states_part)  # (64,196,512)
        hidden_states_part = torch.cat((emb, hidden_states_part), axis=1)  # (64,197,512)
        hidden_states_whole = hidden_states_whole.permute(0, 2, 1).contiguous()  # (64,197,512)
        hidden_states = hidden_states_part + hidden_states_whole

        if temporal_features is not None and temporal_features.shape[1] == self.mamba_interval:
            hidden_states = self.dstf(hidden_states, temporal_features)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            "mixer_part": self.mixer_part.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            ),
            "mixer_whole": self.mixer_whole.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            ),
        }


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class HiSTF_Mamba(nn.Module):
    def __init__(
        self,
        depth=24,
        embed_dim=512,
        d_state=16,
        ssm_cfg=None,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
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
        dataset="humanml",
        mamba_interval=3,
        **kwargs,
    ):
        """Init."""
        factory_kwargs = {"device": device, "dtype": dtype}

        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.mamba_interval = mamba_interval

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.layers = nn.ModuleList()
        self.mamba_interval = mamba_interval
        for i in range(depth):
            self.layers.append(
                create_block(
                    d_model=embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
            )

            if (i + 1) % self.mamba_interval == 0:
                self.layers.append(
                    Daul_Spatial_Mamba(
                        d_state=d_state,
                        layer_idx=i + 1,
                        bimamba_type=bimamba_type,
                        if_divide_out=if_divide_out,
                        init_layer_scale=init_layer_scale,
                        residual_in_fp32=residual_in_fp32,
                        ssm_cfg=ssm_cfg,
                        factory_kwargs=factory_kwargs,
                        norm_cls=nn.LayerNorm,
                        drop_path=drop_path_rate,
                        dataset=dataset,
                        mamba_interval=self.mamba_interval,
                    )
                )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.temporal_features = None

        self.dataset = dataset

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache."""
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed",
            "cls_token",
            "dist_token",
            "cls_token_head",
            "cls_token_tail",
        }

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, x, text=None, inference_params=None):
        """Forward."""

        hidden_states = self.forward_features(x, inference_params)

        return hidden_states

    def forward_features(self, x, inference_params=None):
        B, T, C = x.shape
        hidden_states = x
        residual = None
        temporal_features_list = []
        temporal_layer_count = 0
        temporal_features = None

        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, Daul_Spatial_Mamba):
                temporal_layer_count += 1
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
                temporal_features_list.append(hidden_states)

                if (
                    temporal_layer_count < (self.mamba_interval + 1)
                    and len(temporal_features_list) == self.mamba_interval
                ):
                    temporal_features = torch.stack(temporal_features_list, dim=1)
                    temporal_features_list = []
                    temporal_layer_count = 0
            else:
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                    temporal_features=temporal_features,
                )

        return hidden_states
