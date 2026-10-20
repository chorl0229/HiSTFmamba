---
language: en
license: mit
tags:
  - text-to-motion
  - mamba
  - diffusion
---
# HiSTFmamba model card

Associated paper: [Linear Multi-granular Body Modeling for Text-Driven Motion Generation (accepted at ICME 2026)](https://arxiv.org/abs/2503.06897).
Current maintenance checks: [validation](docs/MAINTENANCE_VALIDATION.md).

## Status

**No verified pretrained HiSTF checkpoint is included.** This card documents a source-release candidate and the evidence needed to complete its weight release.

## Model and intended use

A text-conditioned motion diffusion framework combining bidirectional temporal scans, anatomical part and whole-body spatial scans, and dynamic spatiotemporal fusion. Intended for research on 3D human motion synthesis and animation. HumanML3D uses 263 features / 22 joints; KIT-ML uses 251 features / 21 joints.

## Provenance

The release uses surviving January 2026 anatomical HiSTF/DSFM modules and the pre-RDDM Lightning diffusion wrapper. Later EC-DynaPart, LQ, and residual-diffusion experiments are excluded. The selected six-block recipe is an interpretation of manuscript N=3, L=2, with unresolved parameter-count and complexity differences.

## Weight search

The search identified 213 candidate files and examined all 149 Lightning checkpoints for the archived DSFM parameter names. None matched. Sixteen representative training directories were inspected at tensor level and contained EC-DynaPart variants. No such weights were relabeled as HiSTF.

## Evaluation

The manuscript reports FID 0.039 (HumanML3D), FID 0.217 (KIT-ML), and 23.95M parameters. The reconstructed HumanML3D denoiser has 15,534,923 trainable parameters. CPU component tests pass; these are not benchmark reproduction. The full CUDA path has not been executed in this environment.

## Required evidence for a complete release

Recover the final checkpoint, its exact model source/configuration, normalization statistics, and the evaluation settings/log associated with the paper table. Require strict denoiser compatibility and rerun evaluation before advertising the weights as reproducing the paper.

## License and hosting

Author-owned code and future author-owned weights use MIT. Upstream components and datasets retain their terms. No public weight URL exists in this preparation.
