"""Run CUDA forward/backward through the reconstructed HiSTF architecture."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.nets.light_final import LightT2M

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_dim", type=int, choices=[251, 263], default=263)
parser.add_argument("--num_layers", type=int, default=6)
args = parser.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; no GPU test was run.")
model = LightT2M(input_dim=args.input_dim, num_layers=args.num_layers).cuda().train()
motion = torch.randn(2, 196, args.input_dim, device="cuda")
mask = torch.ones(2, 196, device="cuda", dtype=torch.bool)
output = model(
    motion,
    mask,
    torch.tensor([500, 250], device="cuda"),
    {"text_emb": torch.randn(2, 512, device="cuda")},
)
assert output.shape == motion.shape and torch.isfinite(output).all()
output.square().mean().backward()
assert torch.isfinite(model.input_process.poseEmbedding.weight.grad).all()
print("CUDA forward and backward passed; this does not establish paper-checkpoint compatibility.")
