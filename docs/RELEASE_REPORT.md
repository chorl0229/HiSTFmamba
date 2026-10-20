# HiSTFmamba local release preparation report

> Historical preparation record retained from the supplied repository. Current
> maintenance checks are recorded in [MAINTENANCE_VALIDATION.md](MAINTENANCE_VALIDATION.md).

## Deliverables and layout

This method has an independent Git repository with English documentation and comments, source provenance, a model card, citation metadata, MIT notices with upstream exceptions, portable entry points, CPU tests, GPU smoke tools, and a GitHub Actions workflow.

All method-specific files are inside this directory:

- `.git/`: the local repository on `main`, with curated source staged and no commit or remote.
- `docs/`: paper-to-code mapping, source audit, validation results, and checkpoint search evidence.
- `tools/prepare_archives.py`: the independent local packaging script.
- `deps/`: ignored local evaluator dependencies; raw datasets are not included.
- `weights/manifest.json`: explicit missing-checkpoint status and author-owned weight licensing.
- `release_bundles/`: source ZIP, checksums, and archive validation records; ignored by Git.

The parent GitHub directory contains only the two method directories, T2Mmamba and HiSTFmamba. This repository's packaging script has no dependency on the sibling repository. No matching method checkpoint or weight archive is available.

## Source selection

The release retains anatomical part/whole spatial modeling and DSFM from the early source, coupled with the pre-RDDM diffusion wrapper. Later LQ/RDDM, EC-DynaPart, and unrelated experiment branches were excluded. Source timestamps and saved options were used as evidence, with tensor names and shapes taking precedence over filenames and newest-file timestamps. Per-file original and final SHA-256 values are recorded in `source_manifest.json`.

The original causal-conv1d 1.0.0 bundle did not match the selected scan API. This source release vendors official causal-conv1d v1.1.1 at commit `97711965f6ea028d9fd4fe94e4e490b7c1011e11`. Original research directories were not modified.

## Checkpoint search

The search covered research directories under the author's home directory and found 213 candidate weight/checkpoint files, including 149 Lightning checkpoints. The per-checkpoint DSFM/EC-DynaPart search record is in [checkpoint search](checkpoint_search.json). None of those 149 files contained the archived DSFM state names. Representative tensor inspections from 16 `third_new` training directories identified EC-DynaPart variants. Evaluator networks were kept distinct from method weights.

The two large ZIP archives inspected contained no checkpoint files. A legacy StableMoFusion tar archive contains an older T2M checkpoint and evaluator assets, recorded in [legacy archive search](legacy_archive_search.json). Environment and dataset archives were not treated as method checkpoints. No later-method checkpoint was relabeled as HiSTF.

## Unresolved paper evidence

The manuscript's N=3, L=2 / 23.95M model cannot yet be tied to a saved source/checkpoint pair. The surviving four-temporal-block version has 10,113,833 parameters; the documented six-block interpretation has 15,534,923. The archived DSFM constructs a T-by-T temporal attention matrix, and the saved dynamic-convolution mixture uses four kernels. These differ from manuscript descriptions and remain explicitly documented.

Recover the final training source, configuration, checkpoint, normalization arrays, and evaluation log before presenting this package as a complete reproduction of the HiSTF paper.

## Validation and limitations

- All six CPU numerical/loading tests pass.
- Ruff critical checks and formatting, English/path/syntax checks, and CLI/config composition checks pass.
- Both dataset configurations instantiate successfully.
- The source ZIP was extracted into an independent temporary directory and passed CPU tests, source checks, and Ruff checks without working-tree runtime assets.
- Every packaged source file matches its recorded SHA-256, the dataset loader is present, and source archive membership matches the staged Git files.

Local archive validation records are under `release_bundles/` at the repository root. CUDA execution and benchmark reproduction were not possible because the NVIDIA driver was unavailable. Paper metrics are not claimed as reproduced.

## Packaging and publication

Source packages exclude raw datasets, evaluator binaries, checkpoints, logs, build artifacts, caches, and release bundles. Author-owned code and future author-owned weights use MIT; upstream licenses and attribution remain intact.

The user requested local preparation only. No GitHub or model-hub upload was performed. See [release procedure](RELEASE.md) for rebuilding packages and researched tool references.
