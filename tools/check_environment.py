"""Check that this repository's custom backend is selected."""

import inspect
import json
from pathlib import Path

import torch
import causal_conv1d
import mamba_ssm
from mamba_ssm.modules.mamba_simple import Mamba

root = Path(__file__).resolve().parents[1]
backend = Path(mamba_ssm.__file__).resolve()
expected = root / "third_party" / "mamba"
local = expected in backend.parents
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_available": torch.cuda.is_available(),
            "mamba_source": str(backend),
            "local_backend": local,
        },
        indent=2,
    )
)
if not local:
    raise SystemExit("Install the vendored backend with bash tools/install_backend.sh.")

if causal_conv1d.__version__ != "1.1.1":
    raise SystemExit("This backend requires the vendored causal-conv1d 1.1.1 interface.")
