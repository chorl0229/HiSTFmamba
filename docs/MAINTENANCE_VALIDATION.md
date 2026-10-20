# Maintenance validation

Date: 2026-10-20. Scope: accepted-paper metadata, reader documentation, and release tooling.

## Environment

macOS 26.6.2 / arm64; Python 3.9.6; PyTorch 2.2.2; NumPy 1.26.4;
Ruff 0.14.2; cffconvert 2.0.0. Dependencies were installed in a separate temporary
environment. This CPU environment is distinct from the earlier CUDA checkpoint-preparation
environment described in [the historical record](VALIDATION.md).

## Checks executed

```bash
python tools/maintain.py
python tools/prepare_archives.py --source-only
```

The source-only archive was extracted into an independent temporary directory, with
no datasets, evaluator binaries, checkpoints, or installed custom Mamba backend.
`python tools/maintain.py` passed from that extracted directory:

- Ruff critical lint and formatting checks.
- Python syntax, English source text, and portable-path checks across 115 Python/YAML files.
- Publication synchronization and CFF 1.2.0 schema validation.
- Local Markdown file links and heading anchors in 15 maintained documents.
- 6 existing CPU numerical/loading tests.
- 6 maintenance regressions: citation drift, broken links, runtime/symlink exclusion,
  repository ignore rules, weight path containment, and a source-only build with absent weights.

All 180 extracted source files matched archive membership and the per-file source
manifest (the manifest itself is included but does not hash itself). Standalone source
checks therefore passed without hidden working-tree runtime files. Local archive check
logs and checksums are kept in `release_bundles/source-only/`.

No matching HiSTF checkpoint is available, so no method-weight load or export was validated in this refactor.

## Compatibility and limits

Model implementations, legacy import paths, Hydra model targets, checkpoint keys,
and runtime dependency versions were preserved. Documentation training examples for
T2M use new experiment names to avoid overwriting prepared inference exports.

CUDA compilation, GPU forward/backward, end-to-end generation, training, and benchmark
reproduction were not run on this Mac. The CUDA setup recipe and GPU entry points
remain documented; CPU and archive checks do not establish their success.
This record describes the local pre-publication checks. Current remote checks are
available on [GitHub Actions](https://github.com/chorl0229/HiSTFmamba/actions/workflows/checks.yml).
Model weights are distributed separately from the Git source tree.

## Bibliographic verification

`publication.json` records the verified title, ordered authors, venue, year, status,
and evidence URLs. `tools/sync_publication.py` keeps the README paper header,
README BibTeX block, `CITATION.bib`, and `CITATION.cff` consistent.
See [the design references](MAINTENANCE_REFERENCES.md) for the primary sources used.
