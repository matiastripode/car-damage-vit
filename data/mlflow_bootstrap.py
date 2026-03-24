import json
import math
import os
import tempfile
from io import BytesIO
from pathlib import Path

import mlflow
from PIL import Image


DATA_ROOT = Path("/data/raw")
ANN_ROOT = DATA_ROOT / "annotations"
SPLITS = ["train", "validation", "test"]
# Permite reutilizar este script tanto en Docker (sqlite local) como desde pipeline local.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////mlruns/mlflow.db")
EXPERIMENT = os.getenv("MLFLOW_DATASET_EXPERIMENT", "cardd_dataset_inspection")
SAMPLE_RATIO = 0.10
DATASET_VERSION = os.getenv("DATASET_VERSION", "unknown")


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _find_sample_images(path: Path, limit: int | None = None):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    if not path.exists():
        return out
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
            if limit is not None and len(out) >= limit:
                break
    return out


def _sample_size(total: int, ratio: float = SAMPLE_RATIO) -> int:
    if total <= 0:
        return 0
    return max(1, math.ceil(total * ratio))


def _extract_samples_from_hf_split(split_dir: Path, split: str, limit: int):
    try:
        from datasets import load_from_disk
    except Exception:
        return []

    try:
        ds = load_from_disk(str(split_dir))
    except Exception:
        return []

    out = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        n = min(limit, len(ds))
        for i in range(n):
            sample = ds[i]
            img = sample.get("image")
            if img is None:
                continue

            # HuggingFace Image feature suele venir como PIL.Image, pero puede
            # llegar como dict con bytes/path según config de decoding.
            if isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            elif isinstance(img, dict) and img.get("bytes") is not None:
                pil_img = Image.open(BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, dict) and img.get("path"):
                pil_img = Image.open(img["path"]).convert("RGB")
            else:
                continue

            out_file = td_path / f"{split}_{i:03d}.jpg"
            pil_img.save(out_file, format="JPEG", quality=90)
            out.append(out_file)

        if out:
            mlflow.log_artifacts(str(td_path), artifact_path=f"samples/{split}")

    return out


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="docker_startup_dataset_bootstrap"):
        mlflow.log_param("data_root", str(DATA_ROOT))
        mlflow.log_param("annotations_root", str(ANN_ROOT))
        mlflow.log_param("dataset_version", DATASET_VERSION)

        for split in SPLITS:
            split_dir = DATA_ROOT / split
            ann_file = ANN_ROOT / f"{split}.json"

            mlflow.log_param(f"{split}_exists", split_dir.exists())
            mlflow.log_param(f"{split}_file_count", _count_files(split_dir))
            mlflow.log_param(f"{split}_annotation_exists", ann_file.exists())

            if ann_file.exists():
                with open(ann_file, "r") as f:
                    ann = json.load(f)
                mlflow.log_dict(ann, f"annotations/{split}.json")
                mlflow.log_param(f"{split}_images_count", len(ann.get("images", [])))
                mlflow.log_param(f"{split}_annotations_count", len(ann.get("annotations", [])))

            samples = _find_sample_images(split_dir)
            sample_limit = 0
            if samples:
                sample_limit = _sample_size(len(samples))
                samples = samples[:sample_limit]
            else:
                try:
                    from datasets import load_from_disk
                    ds = load_from_disk(str(split_dir))
                    sample_limit = _sample_size(len(ds))
                except Exception:
                    sample_limit = 0

            if samples and sample_limit > 0:
                mlflow.log_param(f"{split}_sample_images_found", len(samples))
                mlflow.log_param(f"{split}_first_sample", str(samples[0]))
                for sample in samples:
                    mlflow.log_artifact(str(sample), artifact_path=f"samples/{split}")
            else:
                hf_samples = _extract_samples_from_hf_split(split_dir, split, limit=sample_limit)
                mlflow.log_param(f"{split}_sample_images_found", len(hf_samples))
                if hf_samples:
                    mlflow.log_param(f"{split}_first_sample", hf_samples[0].name)

        with tempfile.TemporaryDirectory() as td:
            summary = {
                "data_root": str(DATA_ROOT),
                "splits": SPLITS,
                "notes": "Run generado al iniciar docker-compose para inspección inicial del dataset.",
            }
            out = Path(td) / "dataset_summary.json"
            out.write_text(json.dumps(summary, indent=2))
            mlflow.log_artifact(str(out), artifact_path="meta")


if __name__ == "__main__":
    try:
        main()
        print("MLflow bootstrap: dataset registrado correctamente.")
    except Exception as e:
        print(f"MLflow bootstrap: error no bloqueante: {e}")
