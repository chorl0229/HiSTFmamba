# Training, evaluation, and generation

[Back to README](../README.md) · [Setup](SETUP.md)

Run all commands from the repository root.

## Training recipe

The default configuration interprets manuscript N=3, L=2 as six temporal blocks with a spatial/fusion block after every three. This is a documented reconstruction, not a configuration recovered from a confirmed paper checkpoint. The saved four-temporal-block source variant can be selected with `model.denoiser.num_layers=4`.

```bash
python src/train.py data=hml3d_light_final trainer.devices=2 trainer.precision=bf16-mixed trainer.max_epochs=350
python src/train.py data=kit_light_final trainer.devices=2 trainer.precision=bf16-mixed trainer.max_epochs=350
```

The retained diffusion path predicts clean motion (`sample`) with 1,000 training steps and AdamW at 1e-4. Sampling uses 20 UniPC steps; the wrapper converts predicted clean motion to epsilon before the epsilon-configured sampler.
The reconstructed model has 15,534,923 trainable parameters for HumanML3D, whereas the manuscript reports 23.95M. This discrepancy remains unresolved.

## Evaluation and generation after checkpoint recovery

These commands require a checkpoint with an exactly matching denoiser. Loading fails on missing, unexpected, or shape-mismatched denoiser parameters.

```bash
python src/eval.py data=hml3d_light_final ckpt_path=checkpoints/hml3d.ckpt model.metrics.enable_mm_metric=true
python src/sample_motion.py ckpt_path=checkpoints/hml3d.ckpt text="A person walks forward." length=100 repeats=1
# KIT-ML generation additionally selects its normalization directory:
python src/sample_motion.py data=kit_light_final data_dir=./data/KIT-ML ckpt_path=checkpoints/kit.ckpt length=100 repeats=1
```

The spatial scan has 197 fixed channels: one condition token and 196 motion frames. Short generation requests are padded inside the denoiser and cropped on output. Sequences longer than 196 frames are rejected. Output joint arrays have 22 joints for HumanML3D or 21 for KIT-ML.

For reproducible sampling, add `seed=0`; use `save_path=./visual_datas/` to select the output directory. Joint arrays are written under `gen_joints/`. Training, evaluation, and sampling require the same denoiser configuration as the checkpoint.
