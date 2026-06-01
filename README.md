# Reproducible OCR Pipeline with DVC and MinIO

This project is a reproducible preprocessing pipeline for OCR workflows. It extracts page images from mixed raw document inputs, predicts document orientation, rotates images upright, and versions datasets/artifacts with DVC.

The default pipeline uses a deterministic `mock` orientation backend so the repository can run immediately. For production use, add the Paddle orientation model with DVC under `models/doc_ori/PP-LCNet_x1_0_doc_ori` and switch `params.yaml` to `orientation.backend: paddle`.

## Project Layout

```text
.
├── data/
│   ├── raw/mixed/                 # DVC-tracked sample/raw inputs
│   └── interim/                   # DVC pipeline outputs
├── models/doc_ori/                # Real model artifact location
├── src/stages/
│   ├── extract_images.py
│   └── orient_and_rotate.py
├── tests/
├── dvc.yaml
├── params.yaml
├── docker-compose.minio.yml
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requirement files are nested by use case:

- `requirements.txt`: default local development install
- `requirements/base.txt`: preprocessing and DVC only
- `requirements/dev.txt`: tests
- `requirements/train.txt`: EfficientNet orientation training
- `requirements/paddle.txt`: Paddle orientation backend
- `requirements/all.txt`: all optional stacks

Start local MinIO when you want remote DVC storage:

```powershell
Copy-Item .env.example .env
docker compose -f docker-compose.minio.yml up -d
```

Open the MinIO console at <http://127.0.0.1:9001>. Create the `ocr-pipeline` bucket, or use the integration test commands as a reference for creating it with `mc`.

Configure DVC for MinIO:

```powershell
dvc remote add -d minio s3://ocr-pipeline
dvc remote modify minio endpointurl http://127.0.0.1:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin123
```

## Run the Pipeline

```powershell
dvc repro
```

Manual stage commands:

```powershell
python src/stages/extract_images.py --input data/raw/mixed --output data/interim/images_flat
python src/stages/orient_and_rotate.py --input data/interim/images_flat --output data/interim/orientation --model-dir models/doc_ori/PP-LCNet_x1_0_doc_ori --backend mock
```

Outputs:

- `data/interim/images_flat/manifest.jsonl`
- `data/interim/orientation/upright/`
- `data/interim/orientation/manifest.jsonl`
- `data/interim/orientation/preds.csv`

## Add Real Data and Model Artifacts

Place raw inputs in `data/raw/mixed`, then version them:

```powershell
dvc add data/raw/mixed
git add data/raw/mixed.dvc data/raw/.gitignore
dvc push
```

Place the Paddle model at `models/doc_ori/PP-LCNet_x1_0_doc_ori`, then version it:

```powershell
dvc add models/doc_ori/PP-LCNet_x1_0_doc_ori
git add models/doc_ori/PP-LCNet_x1_0_doc_ori.dvc models/doc_ori/.gitignore
dvc push
```

Switch `params.yaml` to:

```yaml
orientation:
  backend: paddle
```

The Paddle backend expects PaddleClas and a compatible Paddle runtime:

```powershell
pip install -r requirements/paddle.txt
```

## Orientation Evaluation Data

Download a bounded Persian/Arabic orientation-evaluation subset:

```powershell
python src/data/download_orientation_eval.py --max-bytes 1073741824 --per-source-limit 40
dvc add data/raw/orientation_eval
```

The downloader targets:

- IDPL-PFOD Persian/Farsi samples from GitHub
- OpenITI MAKHZAN page images via ranged downloads from Zenodo
- Arabic Documents OCR images from Kaggle's public dataset archive

It writes source images plus synthetic rotations under `data/raw/orientation_eval`, with labels in `labels.csv`.
If a network download times out after files have landed, rebuild manifests without downloading again:

```powershell
python src/data/download_orientation_eval.py --from-existing-only
```

Add or resume a table-heavy invoice subset while keeping generated rotations compact:

```powershell
python src/data/download_orientation_eval.py --tables-only --table-target 1000 --max-bytes 1073741824 --rotation-format jpg --jpeg-quality 82 --max-rotated-side 1600
dvc add data/raw/orientation_eval
```

## Train an Orientation Model

This project includes a DVC training stage adapted from [duartebarbosadev/deep-image-orientation-detection](https://github.com/duartebarbosadev/deep-image-orientation-detection/). It fine-tunes an EfficientNetV2-S classifier on upright source images, generating `0`, `90`, `180`, and `270` degree rotations during training.

The default warm-start checkpoint is DVC-managed at:

```text
models/doc_ori/deep_image_orientation_detection/orientation_model_v2_0.9882.pth
```

It comes from [DuarteBarbosa/deep-image-orientation-detection on Hugging Face](https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection), which reports 98.82% validation accuracy.

The stage defaults to `train_orientation.mode: skip` so the normal pipeline remains lightweight. To train:

```powershell
pip install -r requirements/train.txt
```

Edit `params.yaml`:

```yaml
train_orientation:
  mode: train
```

Then run:

```powershell
dvc repro train_orientation_model
```

Outputs are written to `models/doc_ori/deep_orientation` and tracked by DVC. The stage writes `best_model.pth`, `checkpoint.pth`, `metrics.json`, `train_val_split.json`, and label-map files when training runs.

## Tests

Fast tests:

```powershell
pytest
```

Full DVC + MinIO integration test:

```powershell
$env:RUN_DVC_MINIO_TESTS = "1"
pytest tests/test_dvc_minio_integration.py
```
