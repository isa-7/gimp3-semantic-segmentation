import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(
    r"G:\GIMP_ML\GIMP3-ML-GIMP3-ML\GIMP3-ML-GIMP3-ML\gimpml"
)
CHECKPOINT = (
    SOURCE_ROOT
    / "weights"
    / "semseg"
    / "dpt_hybrid-ade20k-53898607.pt"
)

# Import the original DPT implementation without modifying its source.
sys.path.insert(0, str(SOURCE_ROOT / "tools" / "semseg" / "DPT"))

from dpt.models import DPTSegmentationModel


def main():
    print("PyTorch:", torch.__version__)
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint exists:", CHECKPOINT.is_file())

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    print("Building original DPT-Hybrid architecture...")
    model = DPTSegmentationModel(
        num_classes=150,
        path=None,
        backbone="vitb_rn50_384",
        features=256,
        readout="project",
    )

    print("Loading original checkpoint on CPU...")
    checkpoint = torch.load(
        str(CHECKPOINT),
        map_location=torch.device("cpu"),
    )

    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        checkpoint = checkpoint["model"]

    result = model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print("Checkpoint load: SUCCESS")
    print("Strict state-dict match:", result)
    print("Model eval mode:", not model.training)
    print("DPT-Hybrid checkpoint test: PASSED")


if __name__ == "__main__":
    main()
