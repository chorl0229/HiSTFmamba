# Recorded validation

> Historical preparation record retained from the supplied repository. Current
> maintenance checks are recorded in [MAINTENANCE_VALIDATION.md](MAINTENANCE_VALIDATION.md).

Preparation date: 2026-10-20.

## Local environment

Python 3.9.21; PyTorch 2.0.1+cu118; NumPy 1.26.4; Diffusers 0.33.1; timm 1.0.15; Ruff 0.14.2.
The vendored Python Mamba implementation was selected with `PYTHONPATH=third_party/mamba` for model-construction/checkpoint checks. Existing local CUDA extension binaries were importable; rebuilding those binaries was not tested.

## Passed checks

- Ruff formatting and critical static checks pass.
- Standalone source validation passes for 109 Python/YAML files, including vendored source.
- 6 numerical/loading regression tests pass with `python -m unittest discover -s tests -v`.
- HumanML3D and KIT-ML Hydra configurations compose and instantiate the denoiser.
- Training, evaluation, and sampling configurations compose successfully; preprocessing help executes.
- Anatomical reconstruction, dynamic-convolution gradients, DSFM gradients, and strict-denoiser loading tests pass.
- See `model_construction.json` for architecture counts; no paper checkpoint was available to validate.

## Not executed

CUDA denoiser forward/backward, end-to-end motion generation, full training, and dataset-level evaluation were not run. `torch.cuda.is_available()` returned false, and `nvidia-smi` could not communicate with the NVIDIA driver. These results are not a paper-metric reproduction.

The CUDA smoke script and README evaluation commands are provided for a working GPU environment. The installation recipe records relevant dependency versions but is not a fully rebuilt environment lock.
