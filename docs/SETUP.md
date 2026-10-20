# Installation and data preparation

[Back to README](../README.md)

## Get the code

```bash
git clone https://github.com/chorl0229/HiSTFmamba.git
cd HiSTFmamba
```

## Environment

Use a separate environment for each repository: their vendored `mamba_ssm` implementations differ.
The reference versions below come from the earlier checkpoint-preparation environment recorded in [historical validation](VALIDATION.md). They are not a newly rebuilt environment lock.
CUDA compilation and end-to-end generation still need validation on a machine with a working NVIDIA driver.

```bash
conda create -n histfmamba python=3.9 -y
conda activate histfmamba
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
bash tools/install_backend.sh
```

A CUDA 11.8 toolkit with `nvcc`, a compatible C++ compiler, and a working NVIDIA GPU are required by the legacy scan extensions.
The backend installer forces a source build of causal-conv1d 1.1.1 and the custom Mamba implementation.
Do not replace it with an unrelated PyPI `mamba-ssm` wheel.
Run commands from this repository's root. After installation, run `python tools/check_environment.py` to confirm that the correct custom backend is selected.

## Data and evaluation assets

Follow [HumanML3D and KIT-ML](https://github.com/EricGuo5513/HumanML3D) for dataset preparation and [Light-T2M](https://github.com/qinghuannn/light-t2m) for the GloVe and text-motion evaluator assets.

```text
data/
  HumanML3D/{new_joint_vecs/,texts/,Mean.npy,Std.npy,train.txt,val.txt,test.txt}
  KIT-ML/{new_joint_vecs/,texts/,Mean.npy,Std.npy,train.txt,val.txt,test.txt}
deps/
  glove/
  t2m_guo/{t2m,kit}/text_mot_match/model/finest.tar
```

```bash
python src/tools/data_preprocess.py --dataset hml3d --data_dir ./data/HumanML3D
python src/tools/data_preprocess.py --dataset kit --data_dir ./data/KIT-ML
```

Preprocessing produces `data_train.npy`, `data_val.npy`, and `data_test.npy` for the datamodule. Use `paths.data_dir=/path/to/datasets` to relocate data. Evaluator binaries are not included in the source archive.
