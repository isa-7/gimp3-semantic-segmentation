import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "weights"
def prepare_cuda_dlls():
    """Register CUDA/cuDNN DLLs before importing ONNX Runtime."""
    dll_handles = []

    config_path = Path(__file__).resolve().parents[2] / "local-config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Local configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        local_config = json.load(handle)

    dpt_gpu_python = Path(local_config["python"]["dpt_gpu"])
    torch_lib = (
        dpt_gpu_python.parent.parent
        / "Lib"
        / "site-packages"
        / "torch"
        / "lib"
    )

    if not torch_lib.is_dir():
        raise FileNotFoundError(
            f"CUDA/cuDNN DLL directory not found: {torch_lib}"
        )

    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

    if hasattr(os, "add_dll_directory"):
        dll_handles.append(os.add_dll_directory(str(torch_lib)))

    return dll_handles

def class_color(class_id):
    x = int(class_id) + 1
    return (
        (x * 37 + 53) % 256,
        (x * 67 + 97) % 256,
        (x * 97 + 193) % 256,
    )

def main():
    import tempfile

    dll_handles = []

    parser = argparse.ArgumentParser(description="SegFormer-B0 semantic segmentation")
    parser.add_argument("--input", type=Path, help="Input image path; defaults to GIMP-ML temp input")
    parser.add_argument("--output", type=Path, help="Output image path; defaults to GIMP-ML temp result")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing model.onnx and preprocessor_config.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device (default: cpu)")
    args = parser.parse_args()
    if args.device == "cuda":
        dll_handles = prepare_cuda_dlls()

    import onnxruntime as ort
    model_dir = args.model_dir.resolve()

    temp_dir = Path(tempfile.gettempdir()) / "gimpml-semseg"
    input_path = args.input if args.input else temp_dir / "input_image.png"
    output_path = args.output if args.output else temp_dir / "result.png"

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    config_path = model_dir / "preprocessor_config.json"
    model_path = model_dir / "model.onnx"

    with config_path.open("r", encoding="utf-8") as f:
        preproc = json.load(f)

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read input image: {input_path}")

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    target_h = int(preproc.get("size", {}).get("height", 512))
    target_w = int(preproc.get("size", {}).get("width", 512))
    resized = cv2.resize(
        image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    pixels = resized.astype(np.float32)

    if preproc.get("do_rescale", True):
        pixels *= float(preproc.get("rescale_factor", 1.0 / 255.0))

    if preproc.get("do_normalize", True):
        mean = np.asarray(preproc["image_mean"], dtype=np.float32)
        std = np.asarray(preproc["image_std"], dtype=np.float32)
        pixels = (pixels - mean) / std

    tensor = np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)

    if args.device == "cuda":
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable in this ONNX Runtime environment."
            )
        session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "CUDA was requested, but the session did not activate CUDAExecutionProvider."
            )
    else:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    print("Requested device:", args.device)
    print("Active session providers:", session.get_providers())
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: tensor})[0]

    class_map = np.argmax(logits[0], axis=0).astype(np.uint8)
    class_map = cv2.resize(
        class_map, (width, height), interpolation=cv2.INTER_NEAREST
    )

    palette = np.asarray([class_color(i) for i in range(150)], dtype=np.uint8)
    mask_rgb = palette[class_map]
    result_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result_bgr):
        raise RuntimeError(f"Could not write output image: {output_path}")

    print("SegFormer-B0 inference succeeded.")
    print("Input:", input_path)
    print("Output:", output_path)
    print("Image size:", width, "x", height)
    print("Detected classes:", len(np.unique(class_map)))

if __name__ == "__main__":
    main()
