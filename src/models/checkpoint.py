"""Load every denoiser parameter strictly while allowing separately supplied CLIP."""

import torch


def load_denoiser_checkpoint(model, path):
    # Native Lightning training checkpoints also serialize scheduler hyperparameters.
    # This entry point accepts a locally supplied, trusted training checkpoint.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["state_dict"].items()}
    denoiser = {k[len("denoiser.") :]: v for k, v in state.items() if k.startswith("denoiser.")}
    if not denoiser:
        raise ValueError("Checkpoint contains no denoiser parameters.")
    model.denoiser.load_state_dict(denoiser, strict=True)
    # Frozen CLIP and evaluator weights may be supplied separately.
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint parameters: {result.unexpected_keys}")
    return checkpoint
