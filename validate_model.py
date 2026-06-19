"""
Validate the best trained model on the val set and print metrics.
Run after training completes.
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO

if not torch.cuda.is_available():
    sys.exit("ERROR: No CUDA GPU detected. Aborting.")

# ── Point to your best model ───────────────────────────────────────────────────
RUN_DIR   = Path(r"C:\Users\lojai\Documents\labelling\runs\tyre_seg")
DATA_YAML = Path(r"C:\Users\lojai\Documents\labelling\yolo_whole_data_split\dataset.yaml")

best = RUN_DIR / "weights" / "best.pt"
if not best.exists():
    # Try to find latest run if tyre_seg doesn't exist yet
    candidates = sorted(Path(r"C:\Users\lojai\Documents\labelling\runs").glob("tyre_seg*/weights/best.pt"))
    if not candidates:
        sys.exit(f"ERROR: No best.pt found. Make sure training has completed.")
    best = candidates[-1]

print(f"Loading model: {best}")
model = YOLO(str(best))

metrics = model.val(
    data    = str(DATA_YAML),
    imgsz   = 640,
    batch   = 8,
    device  = 0,
    plots   = True,
    verbose = True,
)

print("\n── Validation Results ─────────────────────────────────────────")
print(f"  mAP50      (box) : {metrics.box.map50:.4f}")
print(f"  mAP50-95   (box) : {metrics.box.map:.4f}")
print(f"  mAP50      (seg) : {metrics.seg.map50:.4f}")
print(f"  mAP50-95   (seg) : {metrics.seg.map:.4f}")
print(f"  Precision        : {metrics.box.mp:.4f}")
print(f"  Recall           : {metrics.box.mr:.4f}")
