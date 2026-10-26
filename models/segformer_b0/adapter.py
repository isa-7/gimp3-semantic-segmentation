from pathlib import Path
import json
import subprocess
import sys


MODEL_DIR = Path(__file__).resolve().parent / "weights"
INFER_SCRIPT = Path(__file__).resolve().parent / "infer.py"


def run_inference(
    input_path,
    output_path,
    python_executable=None,
    device="cpu",
):
    """Run SegFormer-B0 inference in a separate Python process."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")

    interpreter = python_executable or sys.executable
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Input image not found: {input_file}")

    if not (MODEL_DIR / "model.onnx").is_file():
        raise FileNotFoundError(f"ONNX model not found in: {MODEL_DIR}")

    if not (MODEL_DIR / "preprocessor_config.json").is_file():
        raise FileNotFoundError(
            f"Preprocessor configuration not found in: {MODEL_DIR}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(interpreter),
        str(INFER_SCRIPT),
        "--input", str(input_file),
        "--output", str(output_file),
        "--model-dir", str(MODEL_DIR),
        "--device", device,
    ]

    env = None
    if device == "cuda":
        import os

        env = os.environ.copy()
        config_file = Path(__file__).resolve().parents[2] / "local-config.json"
        if not config_file.is_file():
            raise FileNotFoundError(
                f"Local configuration not found: {config_file}"
            )

        with config_file.open("r", encoding="utf-8") as handle:
            local_config = json.load(handle)

        dpt_gpu_python = Path(local_config["python"]["dpt_gpu"])
        dpt_gpu_lib = (
            dpt_gpu_python.parent.parent
            / "Lib"
            / "site-packages"
            / "torch"
            / "lib"
        )
        if not dpt_gpu_lib.is_dir():
            raise FileNotFoundError(
                f"CUDA/cuDNN DLL directory not found: {dpt_gpu_lib}"
            )
        env["PATH"] = str(dpt_gpu_lib) + os.pathsep + env.get("PATH", "")

    subprocess.run(command, check=True, env=env)
    return output_file
