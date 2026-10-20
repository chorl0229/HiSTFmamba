# Repository maintenance references

Reviewed on 2026-10-20. These are design references, not additional dependencies.

- [MoMask — CVPR 2024](https://github.com/EricGuo5513/momask-codes): put the paper, project identity, environment, generation, and dataset-specific evaluation commands where a new reader can find them.
- [Light-T2M — AAAI 2025](https://github.com/qinghuannn/light-t2m): document dataset preprocessing and the directory layout of evaluator assets separately from model weights.
- [Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model): describe sampling and evaluation options and record changes that affect previously reported results.
- [Motion In-betweening via Two-stage Transformers — ACM TOG 2022](https://github.com/victorqin/motion_inbetweening): connect each runnable command to its configuration, required artifacts, and expected output. [Publisher record](https://doi.org/10.1145/3550454.3555454).
- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist): record exact commands, environments, experimental settings, and the subset actually reproduced. Used as a reproducibility reference, not as an ICRA/ICME submission requirement.
- [GitHub citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files): use `preferred-citation` to direct GitHub's citation UI to the conference paper.

## Decisions used here

The README introduces the accepted paper, gives a short working path, and links to detailed guides. Availability of code, pretrained weights, and reproduced metrics are separate facts. An acceptance announcement does not imply that a checkpoint has been released or tested.

Keep title, ordered authors, venue, year, publication status, and verified URLs in `docs/publication.json`. `tools/sync_publication.py` generates the CFF, BibTeX, and README paper header and citation block. The arXiv DOI is not a proceedings DOI; an absent proceedings DOI remains absent.

Keep experimental implementations and checkpoint keys stable during documentation and tooling refactors. The two repositories intentionally retain separate custom Mamba backends and environments. Detailed source discrepancies remain in `SOURCE_AUDIT.md` and `ARCHITECTURE.md`.

The offline maintenance checks cover documentation links, citation consistency, syntax, formatting, and CPU components. They do not certify CUDA behavior, public download availability, or benchmark scores.
