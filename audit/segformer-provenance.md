# SegFormer-B0 Provenance Audit

Audit date: 2026-10-26
Status: Provenance evidence collected; licensing review remains open.

## Reference source

Hugging Face repository:
https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512

Base model reference:
https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

## Local artifact checksums

- model.onnx:
  3e5c18a4be395f16646438d54c42377ddc202edfa33d5eced0c9506de75c44c2

- preprocessor_config.json:
  dbabd93c735c8a5c39ef207c6c4459bf2d261a5dcc55e1ba1c1b982e5947f518

- labels.json:
  69dcbd2f7721d97f251bdebd9958ffe678764f5f9adb940182f3ac6a6fecfd8b

## Verification results

- The local ONNX hash matches the previously checked Xenova ONNX artifact.
- The local preprocessor_config.json hash matches the downloaded Xenova file.
- Local labels.json contains 150 labels.
- Xenova config.json contains 150 id2label entries.
- Comparison found zero differences between local labels and source id2label.
- Xenova config.json SHA-256:
  435799652b2b64c3e422dea20fed4c59651dae9f0e291fd885e9e067fee0ce2a

## Limitations

These checks support artifact identity and metadata provenance.
They do not establish permission to redistribute the ONNX model,
metadata, labels, or other derived artifacts, and do not determine
whether commercial use is permitted.

Review the applicable upstream licenses and terms before redistribution.
Keep model.onnx excluded from public release until that review is complete.
