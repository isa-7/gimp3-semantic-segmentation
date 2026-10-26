import argparse
import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_FILE = PROJECT_ROOT / "local-config.json"

if not LOCAL_CONFIG_FILE.is_file():
    raise FileNotFoundError(
        f"Local configuration not found: {LOCAL_CONFIG_FILE}"
    )

with LOCAL_CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    LOCAL_CONFIG = json.load(config_file)

SOURCE_ROOT = Path(LOCAL_CONFIG["dpt_source_root"])
DPT_ROOT = SOURCE_ROOT / "tools" / "semseg" / "DPT"
DEFAULT_CHECKPOINT = Path(LOCAL_CONFIG["dpt_checkpoint"])

sys.path.insert(0, str(DPT_ROOT))

from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt_util.pallete import get_mask_pallete


def main():
    parser = argparse.ArgumentParser(
        description="DPT-Hybrid ADE20K semantic segmentation"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device: cpu or cuda",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA-compatible GPU is available."
        )
    device = torch.device(args.device)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    checkpoint_path = args.checkpoint.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DPT-Hybrid checkpoint not found: {checkpoint_path}"
        )

    torch.set_num_threads(4)

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read input image: {input_path}")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    image = image.astype(np.float32) / 255.0

    model = DPTSegmentationModel(
        num_classes=150,
        path=None,
        backbone="vitb_rn50_384",
        features=256,
        readout="project",
    )

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=torch.device("cpu"),
    )
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
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

    input_array = transform({"image": image})["image"]
    sample = torch.from_numpy(input_array).unsqueeze(0)
    sample = sample.to (device)

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
        class_map = prediction.squeeze().cpu().numpy().astype(np.uint8)

    mask_rgb = np.array(
        get_mask_pallete(class_map, "ade20k").convert("RGB")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), result_bgr):
        raise RuntimeError(f"Could not write output image: {output_path}")

    print("DPT-Hybrid inference succeeded.")
    print("Input:", input_path)
    print("Output:", output_path)
    print("Checkpoint:", checkpoint_path)
    print("Image size:", original_shape[1], "x", original_shape[0])
    print("Detected classes:", len(np.unique(class_map)))
    print("Inference seconds:", round(time.time() - started, 2))


if __name__ == "__main__":
    main()
