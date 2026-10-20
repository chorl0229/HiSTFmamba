# Third-party notices

Author-owned contributions and author-owned model weights use the MIT license in `LICENSE`. This does not replace upstream licenses.

- [Light-T2M](https://github.com/qinghuannn/light-t2m): inherited training, data, and evaluation infrastructure. The original MIT copyright notice is retained in `LICENSE`.
- [Vision Mamba](https://github.com/hustvl/Vim): bidirectional Mamba integration and backbone portions. Retained license: `third_party/VISION_MAMBA_LICENSE`. Modifications are documented in `docs/SOURCE_AUDIT.md`.
- [Mamba](https://github.com/state-spaces/mamba): vendored selective-scan implementation, Apache-2.0; see `third_party/mamba/LICENSE` and retained source copyright headers.
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d): vendored CUDA convolution extension v1.1.1 (commit `97711965f6ea028d9fd4fe94e4e490b7c1011e11`), BSD-3-Clause; see `third_party/causal-conv1d/LICENSE`.
- [CLIP](https://github.com/openai/CLIP): frozen text encoder, MIT. No HiSTF method-weight export is currently available; see `third_party/CLIP_LICENSE`.
- [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), and the KIT Motion-Language dataset provide evaluation protocols and data representations. Dataset files, GloVe vectors, and evaluator weights are excluded from source archives and remain subject to their respective terms.
- timm, PyTorch, Lightning, Accelerate, Diffusers, and other installed dependencies keep their upstream licenses.

Third-party binary build products and unrelated image-model projects are not redistributed. No SMPL/SMPL-X model files or motion datasets are included in the prepared source or method-weight archives.
