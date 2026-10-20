# Contributing

Bug reports, documentation improvements, and reproducibility fixes are welcome.
Use the issue templates to include the exact command, environment, dataset, checkpoint
checksum, and traceback. For a result mismatch, also include the split, seed, sampler,
precision, and replication count.

## Development checks

Use the repository's own runtime environment for model work. For CPU-only checks,
install PyTorch and NumPy, then the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python tools/maintain.py
```

For documentation/tooling work without PyTorch, use `python tools/maintain.py --static`.
Run `ruff format .` to format edited Python files. CI runs the complete CPU suite.

## Changes to paper code

- Keep legacy import paths, constructor defaults, tensor layouts, and state-dictionary keys compatible unless the change intentionally requires a new checkpoint.
- Explain any algorithmic change, its effect on existing checkpoints, and the numerical or GPU check that supports it.
- Keep datasets, evaluator binaries, weights, logs, and generated outputs outside the source release. Retain upstream license notices.
- Update the corresponding guide when an entry point or option changes.
- Label paper-reported results and newly measured results separately. Record the command, environment, checkpoint SHA-256, seed, sampler, split, and precision for a new measurement.

## Publication metadata

Edit `docs/publication.json`, then run:

```bash
python tools/sync_publication.py
python tools/sync_publication.py --check
```

This updates `CITATION.cff`, `CITATION.bib`, and the README paper header and citation block.
Use verified conference/publisher/arXiv records and preserve author order.
Do not invent download links, DOI values, pages, release dates, or passing CI badges.

## Preparing a source release

Follow [the release guide](docs/RELEASE.md). Inspect the source archive in a clean
directory and run its checks before publishing. Publication is a separate action;
local packaging does not upload anything.
