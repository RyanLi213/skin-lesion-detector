# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A melanoma triage research prototype. The whole project lives in one notebook,
`notebooks/melanoma_resnet18_pipeline.ipynb`. It trains a small head on a **frozen ResNet18**
backbone in a low-data setting, measures how sensitivity holds up as image quality drops, and
exports an **INT8 ONNX** model for offline Raspberry Pi inference. There is no package, test suite,
linter or build system. The Pi camera/inference kit is a separate follow-on project and is not in
this repo.

## Environment and running

- Python 3.13 venv at `.venv/` (gitignored) with torch, torchvision, onnx, onnxruntime, onnxscript
  and ipykernel. Dependencies aren't pinned in a file. The notebook's first markdown cell lists them:
  `pip install torch torchvision scikit-learn pillow pandas numpy matplotlib onnx onnxruntime onnxscript`
- Run the notebook with the `.venv` kernel. **The working directory must be `notebooks/`**: every
  path in `Config` is relative to it (`data/labels.csv`, `outputs/...`).
- Fetching the ImageNet ResNet18 weights needs internet access the first time.
- nbconvert is not installed. To run headless, install it, then run this from inside `notebooks/`:
  `jupyter nbconvert --to notebook --execute --inplace melanoma_resnet18_pipeline.ipynb`
- To read cell sources without opening Jupyter:
  `.venv/Scripts/python.exe -c "import json;[print(''.join(c['source'])) for c in json.load(open('notebooks/melanoma_resnet18_pipeline.ipynb',encoding='utf-8'))['cells']]"`

## Data

- `Config.data_csv` = `data/labels.csv` (i.e. `notebooks/data/labels.csv`, gitignored), with columns
  `image_path,label[,fitzpatrick]`. The current data is PAD-UFES-20. `image_path` values are
  **absolute paths** to images outside the repo (`C:/Users/li3018/Downloads/imgs_part_{1,2,3}`), so
  `img_dir` stays empty. `label = 1` iff diagnostic is MEL. A `fitzpatrick` of `-1` means unknown.
- If the CSV is missing, `load_records` silently falls back to **synthetic noise images**
  (`use_synthetic_if_missing=True`). Outputs from that run prove nothing, so check which path ran.
- Positives are very rare: about 2.3% prevalence, roughly 10 melanomas in the test split.
  Sensitivity metrics are therefore coarse (each test positive is worth 10 points).
- `split_records` is a seeded (`SEED=42`) shuffle and then a 70/15/15 train/val/test split.

## Pipeline architecture (cells run in order and share global state)

1. **`Config` dataclass (`CFG`)**: the single place for hyperparameters, degradation strengths,
   eval quality levels and output paths.
2. **Degradation model**: `degrade_image(img, quality_drop_pct)` scales JPEG compression, Gaussian
   blur, sensor noise and downsample/upsample linearly from one 0–100 knob. The same function
   powers training augmentation (severity sampled from `train_degrade_min/max_pct`) and the
   evaluation axis, so changing it changes both.
3. **`ResNet18Backbone`**: frozen, `fc = Identity`, gives 512-d embeddings. `backbone.preprocess`
   is set afterwards, from dataset-specific mean/std computed on train images (not ImageNet stats).
   Every other step (training, eval, Grad-CAM, ONNX eval) reuses this preprocess.
4. **`ContrastiveHead`**: holds the only trainable parameters. It has a `projector` (used only for
   the InfoNCE clean-vs-degraded alignment loss) and a `classifier` (one logit). The loss is
   pos-weighted BCE on both the clean and degraded logits, plus `contrastive_weight` × InfoNCE.
   Images are loaded and encoded on the fly for each batch. Embeddings are not cached.
5. **Evaluation**: `evaluate_at_quality_drop` sweeps `CFG.eval_quality_levels` on the test split
   and reports sensitivity/specificity/AUC, with per-Fitzpatrick sensitivity when that column
   exists. "% sensitivity retained" is computed relative to the 0% row.
6. **Grad-CAM** on `layer4`. It registers hooks on `backbone.model` that **are never removed**.
7. **Export**: `DeployableModel` deep-copies the backbone and `head.classifier` (the projector is
   dropped). Before deep-copying or exporting, the GradCAM hooks must be cleared from the source
   modules, or the copy/trace fails; the export cells already do this. The FP32 state_dict leads
   to two artifacts: PyTorch dynamic INT8 (a size check only) and ONNX FP32 → ORT `quantize_dynamic`
   INT8. Keep `torch.onnx.export(..., dynamo=False)` because the dynamo exporter breaks on
   `dynamic_axes`. Opset is 18, the input is `pixel_values`, the output is `logit`.
8. **Deployed-artifact eval**: the same quality sweep runs through the INT8 ONNX session. This is
   the number to quote for deployment claims.

Generated artifacts go to `notebooks/outputs/` and are committed (`.pt`, `.onnx`, plots).

## Project conventions from the notebook

- Treat this as a scaffold, not a validated clinical tool. Present outputs as a triage/referral
  signal, never as a diagnosis.
- Read robustness claims (e.g. "X% sensitivity retained at 60% quality drop") off the measured
  curves. Don't restate them as targets.
- A planned backbone swap is `torchvision.models.mobilenet_v3_small`, but only after the ResNet18
  baseline is validated.
