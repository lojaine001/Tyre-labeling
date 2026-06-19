"""
Train YOLOv8m-seg on the combined tyre dataset.
Requires CUDA — will abort if no GPU is detected.
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO


def main():
    # ── GPU guard ──────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        sys.exit("ERROR: No CUDA GPU detected. Training requires a GPU. Aborting.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")

    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_YAML = Path(r"C:\Users\lojai\Documents\labelling\yolo_whole_data_split\dataset.yaml")
    RUN_DIR   = Path(r"C:\Users\lojai\Documents\labelling\runs")

    if not DATA_YAML.exists():
        sys.exit(f"ERROR: dataset.yaml not found at {DATA_YAML}\nRun split_dataset.py first.")

    # ── Model ──────────────────────────────────────────────────────────────────
    weights = Path(r"C:\Users\lojai\Documents\labelling\yolov8m-seg.pt\yolov8m-seg.pt")
    if not weights.exists():
        sys.exit(f"ERROR: Model weights not found at {weights}\nDownload from: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-seg.pt")
    model = YOLO(str(weights))

    # ── Training ───────────────────────────────────────────────────────────────
    results = model.train(
        data          = str(DATA_YAML),
        epochs        = 30,
        patience      = 10,
        imgsz         = 800,
        batch         = 8,
        device        = 0,
        workers       = 4,
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 3,
        warmup_momentum = 0.8,
        cos_lr        = True,
        augment       = True,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 10.0,
        translate     = 0.1,
        scale         = 0.5,
        flipud        = 0.5,
        fliplr        = 0.5,
        mosaic        = 1.0,
        close_mosaic  = 10,
        project       = str(RUN_DIR),
        name          = "tyre_seg_800",
        exist_ok      = False,
        save          = True,
        save_period   = 10,
        val           = True,
        plots         = True,
        verbose       = True,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best model : {best}")
    print(f"Results    : {results.save_dir}")


if __name__ == "__main__":
    main()
