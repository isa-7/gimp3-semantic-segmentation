from pathlib import Path
import subprocess
import sys

INFER_SCRIPT = Path(__file__).resolve().parent / "infer.py"

import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_FILE = PROJECT_ROOT / "local-config.json"

if not LOCAL_CONFIG_FILE.is_file():
    raise FileNotFoundError(
        f"Local configuration not found: {LOCAL_CONFIG_FILE}"
    )

with LOCAL_CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    LOCAL_CONFIG = json.load(config_file)

DEFAULT_CHECKPOINT = Path(LOCAL_CONFIG["dpt_checkpoint"])


def run_inference(
    input_path,
    output_path,
    python_executable=None,
    checkpoint_path=None,
    device="cpu",
):
    """Run DPT-Hybrid inference on CPU or CUDA."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")

    interpreter = python_executable or sys.executable
    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT).resolve()
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Input image not found: {input_file}")
    if not INFER_SCRIPT.is_file():
        raise FileNotFoundError(f"DPT inference script not found: {INFER_SCRIPT}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DPT-Hybrid checkpoint not found: {checkpoint}")
    if not Path(interpreter).is_file():
        raise FileNotFoundError(f"Python interpreter not found: {interpreter}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(interpreter),
        str(INFER_SCRIPT),
        "--input", str(input_file),
        "--output", str(output_file),
        "--checkpoint", str(checkpoint),
        "--device", device,
    ]

    subprocess.run(command, check=True)

    if not output_file.is_file():
        raise RuntimeError(
            f"DPT-Hybrid inference completed without output: {output_file}"
        )

    return output_file
