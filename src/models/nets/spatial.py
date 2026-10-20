"""Dynamic convolutions and anatomical part projections."""

import math

import torch
from torch import nn
import torch.nn.functional as F
from .whole_part import whole2parts


class Attention2D(nn.Module):
    def __init__(self, in_planes, K):
        super(Attention2D, self).__init__()
        self.in_planes = in_planes
        self.K = K

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, kernel_size=1)
        self.fc2 = nn.Conv2d(K, K, kernel_size=1)

    def forward(self, x):
        # x: (batch_size, in_planes, H, W)
        x = self.avgpool(x)  # (batch_size, in_planes, 1, 1)
        x = self.fc1(x)  # (batch_size, K, 1, 1)
        x = F.relu(x)
        x = self.fc2(x)  # (batch_size, K, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, K)
        return F.softmax(x, dim=1)  # (batch_size, K)


class Dynamic_conv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
        dilation=1,
        groups=1,
        bias=True,
        K=8,
    ):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size  # tuple
        self.stride = stride
        self.padding = padding  # tuple
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K

        self.attention = Attention2D(in_planes, K)

        self.weight = nn.Parameter(
            torch.Tensor(K, out_planes, in_planes // groups, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias_param = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        """Forward."""
        softmax_attention = self.attention(x)  # (batch_size, K)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)  # (1, batch_size * in_planes, H, W)

        weight = self.weight.view(self.K, -1)  # (K, out_planes * in_planes/groups * kH * kW)

        aggregate_weight = torch.mm(
            softmax_attention, weight
        )  # (batch_size, out_planes * in_planes/groups * kH * kW)
        aggregate_weight = aggregate_weight.view(
            batch_size * self.out_planes,
            self.in_planes // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        if self.bias_param is not None:
            aggregate_bias = torch.mm(
                softmax_attention, self.bias_param
            )  # (batch_size, out_planes)
            aggregate_bias = aggregate_bias.view(-1)
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class Part_Mamba_Conv_Stage(nn.Module):
    def __init__(self, embed_dim=128, activation="relu", K=8, dataset="humanml"):
        super(Part_Mamba_Conv_Stage, self).__init__()

        self.dataset = dataset

        self.dynamic_conv = Dynamic_conv2d(
            in_planes=6,
            out_planes=6,
            kernel_size=(3, 1),
            padding=(1, 0),
            stride=1,
            groups=6,
            bias=True,
            K=K,
        )

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.parts = ["Root", "R_Leg", "L_Leg", "Backbone", "R_Arm", "L_Arm"]
        self.num_parts = len(self.parts)

        if dataset.lower() == "humanml":
            self.emb_projection = nn.Linear(263, 128)
            self.input_projection = nn.Linear(512, 263)
            dims_dict = {
                "Root": 7,
                "R_Leg": 50,
                "L_Leg": 50,
                "Backbone": 60,
                "R_Arm": 60,
                "L_Arm": 60,
            }
        else:  # kit
            self.emb_projection = nn.Linear(251, 128)
            self.input_projection = nn.Linear(512, 251)
            dims_dict = {
                "Root": 7,
                "R_Leg": 62,
                "L_Leg": 62,
                "Backbone": 48,
                "R_Arm": 48,
                "L_Arm": 48,
            }

        self.projections_in = nn.ModuleList(
            [nn.Linear(dims_dict[part], embed_dim) for part in self.parts]
        )

        self.projections_out = nn.ModuleList(
            [nn.Linear(embed_dim, dims_dict[part]) for part in self.parts]
        )

    def forward(self, x):
        """Forward."""

        x = self.input_projection(x)  # (batch_size, sequence_length, 263)
        emb = x[:, 0:1, :]
        emb = self.emb_projection(emb)
        x = x[:, 1:, :]

        parts = whole2parts(
            x, mode="t2m" if self.dataset == "humanml" else "kit"
        )  # list of 6 tensors, each (batch_size, seq_length, feature_dim_i)

        parts_proj = []
        for i, (proj, part) in enumerate(zip(self.projections_in, parts)):
            projected_part = proj(part)  # (64,196,128)

            projected_part = torch.cat([emb, projected_part], dim=1)

            parts_proj.append(projected_part)

        seq_length = parts_proj[0].size(1)
        stacked = torch.stack(parts_proj, dim=1)  # (batch_size, 6, seq_length, embed_dim)

        stacked = stacked.permute(0, 1, 3, 2).contiguous()  # (batch_size, 6, 128, 197)

        stacked = self.dynamic_conv(stacked)  # (batch_size, 6, 128, 197)

        stacked = self.activation(stacked)  # (batch_size, 6, 128, 197)

        stacked = stacked.contiguous().view(stacked.size(0) * self.num_parts, 128, -1)

        return stacked
