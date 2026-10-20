# Paper-to-code map

The denoiser projects motion features into 512 channels, combines timestep and CLIP text conditions, prepends the condition token, and applies sinusoidal positions.

- **Bi-Temporal Mamba:** `Block` in `histf_mamba.py`, using the vendored bidirectional Mamba implementation.
- **Dual-Spatial Mamba:** `Daul_Spatial_Mamba`, with parallel part and whole-body streams.
- **Part stream:** `whole2parts` forms Root, Right Leg, Left Leg, Backbone, Right Arm, and Left Arm. `Part_Mamba_Conv_Stage` projects them to 128 channels and uses four mixture kernels in the saved dynamic-convolution constructor.
- **Whole-body stream:** scans transposed latent features to preserve global coordination.
- **DSFM:** `DSTM` projects the recent temporal features, forms temporal attention, and fuses the result with the spatial output.
- **Diffusion:** the pre-RDDM Lightning wrapper uses DDPM training, classifier-free guidance, and UniPC sampling.

## Fixed spatial geometry

Spatial scan width is hard-coded to 197 (196 motion frames plus one condition token). The release wrapper pads shorter inputs to this width and crops outputs. `motion_mask` is retained in the denoiser interface; the archived backbone itself does not apply token masking inside every spatial or temporal scan. Loss and sampling masks are handled in the wrapper.

The anatomical split/reconstruction is tested as an identity for both 263-dimensional HumanML3D and 251-dimensional KIT-ML features, including shared-joint averaging.

## Configuration and complexity limits

The manuscript describes N temporal blocks per stage and L stages. The release recipe maps N=3, L=2 to six temporal blocks with two spatial/fusion insertions, using the surviving archived modules. Its parameter count does not match the reported model. The four-temporal-block archive variant remains selectable.

The archived DSFM explicitly builds a `(batch, time, time)` attention matrix. This operation is quadratic in time and must not be advertised as a verified linear-time implementation of the full manuscript. Replacing it with a factorized implementation would change the model and requires author confirmation and checkpoint validation.

Spatial constructor calls set `if_bimamba=True`, which overrides the labels `v3`/`v4` to backend mode `v1`. This effective behavior is preserved. The available backend implements `v1` and `v2`; the labels alone do not identify additional supported scan algorithms.
