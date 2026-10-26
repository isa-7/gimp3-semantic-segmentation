# GIMP3 Semantic Segmentation

An independent development project for semantic segmentation models targeting GIMP 3.2.6.

## Project Status

- Development status: In progress.
- Target GIMP version: 3.2.6.
- Default model: DPT-Hybrid.
- Additional model: SegFormer-B0.
- DPT-Hybrid standalone inference: tested successfully on CPU and CUDA in the local development environment.
- SegFormer-B0 standalone inference: tested successfully on CPU and CUDA in the local development environment.
- GIMP plugin integration: pending verification.
- Clean-environment installation: not yet verified.
- License and redistribution review: in progress.
- SegFormer-B0 ONNX weights and the DPT-Hybrid checkpoint are excluded pending review of their respective provenance and redistribution terms.

Standalone inference results do not establish that the complete plugin works inside GIMP or that the project can be reproduced on another machine without additional setup.

## Project Structure

- `plugin/` — GIMP plugin integration.
- `models/dpt_hybrid/` — DPT-Hybrid adapter and inference implementation.
- `models/segformer_b0/` — SegFormer-B0 adapter and inference implementation.
- `tests/` — test scripts; local input images and generated outputs are excluded from version control.
- `docs/` — technical documentation.
- `configs/plugin.json` — project and plugin metadata.
- `audit/` — dependency snapshots and environment audit files.
- `README.md` — project overview, test status, configuration, and license notes.

## Standalone Inference Tests

The following combinations were tested successfully using a local 1024x1024 test image that is not included in this public repository:

| Model | Device | Result |
| --- | --- | --- |
| DPT-Hybrid | CPU | Passed |
| DPT-Hybrid | CUDA | Passed |
| SegFormer-B0 | CPU | Passed |
| SegFormer-B0 | CUDA | Passed |

The DPT-Hybrid GPU environment used PyTorch 2.0.1 with CUDA 11.8 and an NVIDIA GeForce GTX 1050 Ti.

The SegFormer-B0 GPU environment used ONNX Runtime GPU 1.19.2. The CUDA execution provider was available during validation.

These results apply to the tested local machine and configurations. They do not guarantee compatibility with other hardware, drivers, CUDA installations, or software environments.

Generated PNG outputs in `tests/` are local validation artifacts and are excluded from version control by `.gitignore`.

## Python Environments and Requirements

The project uses separate Python environments to isolate model dependencies.

| Requirements file | Intended environment | Status |
| --- | --- | --- |
| `audit/dpt-cpu-requirements.txt` | DPT-Hybrid CPU | Core dependency list; clean installation not tested |
| `audit/dpt-gpu-requirements.txt` | DPT-Hybrid GPU | Core dependency list; clean installation not tested |
| `audit/segformer-cpu-requirements.txt` | SegFormer-B0 CPU | Core inference dependency list; clean installation not tested |
| `audit/segformer-gpu-requirements.txt` | SegFormer-B0 GPU | Based on the tested environment's installed packages; clean installation not tested |
| `audit/dptenv-requirements.txt` | Historical DPT CPU audit | Retained as an audit record |
| `audit/segformerenv-requirements.txt` | Historical SegFormer CPU audit | Retained as an audit record |

The new requirements files are not yet verified as complete, reproducible installation specifications. Do not assume that installing any one of these files will recreate a working environment from scratch.

### CUDA and GPU Dependencies

The DPT-Hybrid GPU environment was tested with:

- Python 3.10.11
- PyTorch 2.0.1+cu118
- Torchvision 0.15.2+cu118
- CUDA build 11.8

The SegFormer-B0 GPU environment was tested with:

- Python 3.10.11
- ONNX Runtime GPU 1.19.2
- `CUDAExecutionProvider` available

GPU inference also depends on compatible GPU drivers and runtime libraries. Verify compatibility on the target machine before creating or rebuilding an environment.

The official PyTorch CUDA 11.8 wheel index is:

<https://download.pytorch.org/whl/cu118>

Do not replace packages in an already-working environment solely to match an example installation command.

## Local Configuration

1. Copy `local-config.example.json` to `local-config.json`.
2. Replace all example placeholders with valid paths on the target machine.
3. Configure the four Python interpreter paths for the intended CPU and GPU environments.
4. Set `dpt_checkpoint` to the local DPT-Hybrid checkpoint file.
5. Set `dpt_source_root` to the directory containing the DPT source tree, including `tools/semseg/DPT`.

The example configuration contains placeholders and is not ready to run until configured.

The current development setup uses machine-specific paths. Do not publish `local-config.json` or include personal directory paths in public documentation.

## Development Principles

- Keep this project separate from the Remove Background project.
- Preserve DPT-Hybrid as the default model.
- Add and validate models incrementally.
- Keep CPU and GPU dependencies isolated in dedicated environments.
- Do not modify the active installed GIMP plugin until standalone changes and integration tests have been reviewed.
- Review project contents, documentation, dependencies, and applicable licenses before public release.

## License and Redistribution

License review is in progress.

The NVIDIA Source Code License for SegFormer contains restrictions on the use of covered works and derivatives, including non-commercial research or evaluation limitations under Section 3.3. The license file is retained locally while its applicability to the project's components is reviewed.

The license and provenance of project source code, pretrained model weights, ONNX exports, and checkpoints must be assessed separately. The SegFormer-B0 ONNX model is excluded from this repository pending final review of its provenance and redistribution terms. The DPT-Hybrid checkpoint is also not included.

Do not assume that commercial use or redistribution is permitted for any model artifact until its applicable license and provenance have been verified.
