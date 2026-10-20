# Source and checkpoint audit

## Evidence and chronology

- Main manuscript: `paper/2026ICME_HiSTF_Mamba/2026ICME_HiSTF_Mamba.md`.
- Supplement: `paper/2026ICME_HiSTF_Mamba_Supplement/2026ICME_HiSTF_Mamba_Supplement.md`.
- Anatomical HiSTF/DSFM modules survive in the main `src/models/nets` tree with source timestamps on 2026-01-16.
- The current Lightning wrapper and configurations were later changed to RDDM/LQ in February 2026. These do not describe the manuscript's DDPM/UniPC pipeline.
- The pre-RDDM wrapper was recovered from `src/models-o/light_final.py`; the main anatomical HiSTF modules were retained. The `models-o` backbone has a different insertion order, so the whole directory was not copied blindly.
- `models-4`, `histf_mamba_new.py`, and later runs contain EC-DynaPart or other experiments and were excluded.

Timestamps establish a useful relative ordering but may reflect copying. They do not establish the final paper experiment configuration.

## Checkpoint search

The workspace search was expanded beyond T2M to the author's home directory. It identified 213 candidate checkpoint/weight files. All 149 Lightning `.ckpt` files were inspected for archived DSFM state names (`dstf`, `dstm`, `dsfm`); none contained them. Representative checkpoints from 16 directories under `third_new/light_new/logs` were deserialized and inspected: they contained EC-DynaPart state and no DSFM state. Evaluation networks were identified separately.

No candidate was promoted to a HiSTF paper checkpoint. Missing weights are recorded in `weights/manifest.json`. This is evidence of missing matching files in the inspected collection, not proof that the original weights no longer exist elsewhere or under another serialization/name.

## Unresolved paper-to-code differences

| Item | Manuscript | Available code / release treatment |
| --- | --- | --- |
| Best architecture | N=3, L=2 | Saved entry point fixes four temporal blocks. Release recipe interprets this as six blocks / two spatial insertions; verification pending. |
| Parameters | 23.95M | Four-block source: 10,113,833; six-block reconstruction: 15,534,923 (HumanML3D, excluding CLIP/evaluator). |
| Complexity | Linear-time framework | Archived DSFM materializes temporal attention of shape T by T. |
| Dynamic convolution | Describes six learnable kernels | Saved constructor selects K=4 mixture kernels across six anatomical channels. |
| Spatial scan labels | Part / whole scans | `if_bimamba=True` forces backend `v1`, overriding `v3`/`v4` labels. |
| Final weights | Required to reproduce tables | No confirmed matching checkpoint found. |

These are substantive research-version questions. The release does not rewrite the method to conceal them.

## Curation changes

Recovered the DDPM/UniPC wrapper; made constructor arguments match Hydra; passed KIT/HumanML dimensions through part mappings; made data paths portable; padded shorter sequences to the fixed spatial width; reset sampling schedules before reading timesteps; enforced strict denoiser loading; extracted pure-PyTorch DSFM and spatial-convolution modules; translated source documentation; and added CPU tests and explicit CUDA smoke tests.

Original data, checkpoints, and experiment directories remain unchanged. See `source_manifest.json` for provenance. The pure-PyTorch extractions keep parameter names under the existing backbone unchanged.

## Backend compatibility repair

The archived causal-conv1d 1.0.0 exposes four/six-argument CUDA calls, while both selected Mamba scan modules pass five/seven arguments including `seq_idx`. The release vendors official causal-conv1d v1.1.1 at commit `97711965f6ea028d9fd4fe94e4e490b7c1011e11` to match that interface. Author experiment files remain untouched. GPU compilation/execution still requires validation.
