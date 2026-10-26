import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

SOURCE_ROOT = Path(
    r"G:\GIMP_ML\GIMP3-ML-GIMP3-ML\GIMP3-ML-GIMP3-ML\gimpml"
)
DPT_ROOT = SOURCE_ROOT / "tools" / "semseg" / "DPT"
CHECKPOINT = (
    SOURCE_ROOT / "weights" / "semseg"
    / "dpt_hybrid-ade20k-53898607.pt"
)
IMAGE_PATH = Path(r"G:\GIMP_ML\test\gambaruji-semseg.jpg")
OUTPUT_PATH = Path(__file__).resolve().parent / "dpt_hybrid_mask.png"

sys.path.insert(0, str(DPT_ROOT))

from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt_util.pallete import get_mask_pallete


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    torch.set_num_threads(4)
    print("Loading input image...")
    bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    image = image.astype(np.float32) / 255.0

    print("Building DPT-Hybrid...")
    model = DPTSegmentationModel(
        num_classes=150,
        path=None,
        backbone="vitb_rn50_384",
        features=256,
        readout="project",
    )

    print("Loading original checkpoint...")
    checkpoint = torch.load(
        str(CHECKPOINT),
        map_location=torch.device("cpu"),
    )
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    transform = Compose([
        Resize(
            480,
            480,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        PrepareForNet(),
    ])

    sample_np = transform({"image": image})["image"]
    sample = torch.from_numpy(sample_np).unsqueeze(0)

    print("Input shape:", tuple(sample.shape))
    print("Running CPU inference...")
    started = time.time()

    with torch.no_grad():
        output = model(sample)
        prediction = F.interpolate(
            output,
            size=original_shape,
            mode="bicubic",
            align_corners=False,
        )
        prediction = torch.argmax(prediction, dim=1) + 1
        mask = prediction.squeeze().cpu().numpy().astype(np.uint8)

    mask_rgb = np.array(get_mask_pallete(mask, "ade20k").convert("RGB"))
    if not cv2.imwrite(
        str(OUTPUT_PATH),
        cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError(f"Could not save output: {OUTPUT_PATH}")

    print("Inference: SUCCESS")
    print("Original image size:", original_shape)
    print("Mask size:", mask.shape)
    print("Unique class IDs:", np.unique(mask).size)
    print("Elapsed seconds:", round(time.time() - started, 2))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
