"""Numerical regressions for anatomical mappings, dynamic convolution, and DSFM."""

import unittest
import tempfile
from pathlib import Path

import torch

from src.models.nets.whole_part import parts2whole, whole2parts
from src.models.nets.fusion import DSTM
from src.models.nets.spatial import Dynamic_conv2d, Part_Mamba_Conv_Stage
from src.models.checkpoint import load_denoiser_checkpoint


class CoreTests(unittest.TestCase):
    def test_checkpoint_rejects_missing_denoiser_parameters(self):
        model = torch.nn.Module()
        model.denoiser = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.ckpt"
            torch.save({"state_dict": {"denoiser.weight": torch.ones(2, 3)}}, path)
            with self.assertRaises(RuntimeError):
                load_denoiser_checkpoint(model, path)

    def test_checkpoint_allows_separate_frozen_text_encoder(self):
        model = torch.nn.Module()
        model.denoiser = torch.nn.Linear(3, 2)
        model.text_encoder = torch.nn.Linear(5, 3)
        expected = {
            "denoiser.weight": torch.ones(2, 3),
            "denoiser.bias": torch.zeros(2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.ckpt"
            torch.save({"state_dict": expected}, path)
            load_denoiser_checkpoint(model, path)
        self.assertTrue(torch.equal(model.denoiser.weight, expected["denoiser.weight"]))

    def test_anatomical_round_trip(self):
        for mode, dim in (("t2m", 263), ("kit", 251)):
            motion = torch.randn(2, 12, dim, requires_grad=True)
            parts = whole2parts(motion, mode=mode)
            reconstructed = parts2whole(parts, mode=mode)
            self.assertEqual(len(parts), 6)
            self.assertTrue(torch.allclose(motion, reconstructed, atol=1e-6))
            reconstructed.square().mean().backward()
            self.assertTrue(torch.isfinite(motion.grad).all())

    def test_part_projections_for_both_datasets(self):
        for dataset in ("humanml", "kit"):
            module = Part_Mamba_Conv_Stage(dataset=dataset, K=4)
            output = module(torch.randn(2, 17, 512))
            self.assertEqual(output.shape, (12, 128, 17))
            self.assertTrue(torch.isfinite(output).all())

    def test_dynamic_convolution_gradients(self):
        module = Dynamic_conv2d(6, 6, kernel_size=(3, 1), padding=(1, 0), groups=6, K=4)
        x = torch.randn(2, 6, 16, 9, requires_grad=True)
        output = module(x)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_fusion_uses_temporal_information(self):
        module = DSTM(dim=32, mamba_interval=3).eval()
        spatial = torch.randn(2, 12, 32, requires_grad=True)
        temporal = torch.randn(2, 3, 12, 32, requires_grad=True)
        output = module(spatial, temporal)
        self.assertEqual(output.shape, spatial.shape)
        self.assertFalse(torch.allclose(output, module(spatial, torch.zeros_like(temporal))))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(temporal.grad).all())
        self.assertGreater(temporal.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
