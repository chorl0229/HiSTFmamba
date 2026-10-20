# Release procedure

The public source repository is [HiSTFmamba](https://github.com/chorl0229/HiSTFmamba). Source releases exclude datasets, evaluator binaries, and model weights. The steps below describe local validation and packaging for subsequent releases.

1. Run the source/CPU checks listed in README.md and inspect `docs/SOURCE_AUDIT.md`.
2. Install this repository's vendored backend in its own environment and run `tools/check_environment.py`.
3. Run `tools/smoke_cuda.py` with the required arguments, then the documented generation and evaluation commands on licensed data.
4. Record evaluation metrics together with checkpoint SHA-256, sampler, seed, dataset split, and environment. Keep paper-reported results separate from rerun results.
5. Publish only the curated source tree. Git ignores datasets, evaluator dependencies, training outputs, and checkpoints.
6. Host the method-weight archive separately, then add the real download URL to `weights/manifest.json` and README.md. Do not substitute an unrelated model's download link.
7. Update confirmed publication metadata in `docs/publication.json` and run `python tools/sync_publication.py`. Add a repository URL only once the real public destination is known.

This repository is self-contained, including its own `.git`, documentation, tools, and ignored runtime assets. Its local `release_bundles/` directory contains `HiSTFmamba-source.zip`, `SHA256SUMS`, and archive validation records. It is excluded from Git and source archives. There is no method-weight package until the matching checkpoint and architecture are recovered.

Rebuild from this repository's root with `python tools/prepare_archives.py`. The script refreshes `docs/source_manifest.json` and writes only this method's packages. The current empty weight manifest produces a source package only. After verified weights and their checksums are added to `weights/manifest.json`, the script also packages those files. Review and stage the refreshed source manifest with the source changes. Verify archive checksums with `sha256sum --check SHA256SUMS` inside `release_bundles/`.

See [release report](RELEASE_REPORT.md) and [checkpoint search](checkpoint_search.json) for the provenance findings.

## Ongoing maintenance

Run `python tools/maintain.py` before packaging. To build a source-only package
without downloading weights, use:

```bash
python tools/prepare_archives.py --source-only
```

Source-only packages go to `release_bundles/source-only/`, preserving the separate
full release bundle. The source manifest is refreshed for the current working tree,
including new, non-ignored files; it is not a claim that the files have been committed.
Inspect the archive and the Git diff before publishing.

Publication metadata and README conventions follow the
[maintenance references](MAINTENANCE_REFERENCES.md). Current checks are recorded in
[maintenance validation](MAINTENANCE_VALIDATION.md); older preparation reports are historical.
