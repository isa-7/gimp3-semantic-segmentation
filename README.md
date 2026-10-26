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

The original project code is intended to be released under the Apache License, Version 2.0, subject to final review of file provenance and third-party components. See `LICENSE`.

Third-party code and model artifacts may be subject to separate licenses and redistribution conditions. See `THIRD_PARTY_NOTICES.md` and `audit/segformer-provenance.md` for the current attribution and provenance records.

### DPT-Hybrid

The DPT implementation used by this project includes or relies on code from DPT associated with Intel Intelligent Systems Lab. The identified upstream source tree contains an MIT license. See `THIRD_PARTY_NOTICES.md`. The provenance and applicable terms of each redistributed model artifact must be assessed separately.

### SegFormer-B0

The SegFormer-B0 model artifacts were traced to the Xenova repository and the NVIDIA SegFormer model lineage. The NVIDIA SegFormer repository contains a license with non-commercial research and evaluation restrictions.

The precise license applicability to each local source file, metadata file, and model artifact remains under review. The SegFormer-B0 ONNX model is excluded from public redistribution until its applicable terms and permissions have been confirmed.

### Model and checkpoint policy

Do not assume that source-code licensing grants permission to redistribute pretrained weights, ONNX exports, checkpoints, or derived artifacts. Their provenance, license terms, and intended use must be reviewed separately.

This project’s licensing documentation is a work in progress. Please review the applicable third-party terms before commercial use or redistribution.
