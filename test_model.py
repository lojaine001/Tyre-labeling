"""
Test the trained tyre segmentation model with perfect circle post-processing.
After inference, fits a mathematically perfect circle to each predicted mask
so the output is always clean and circular.
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(r"C:\Users\lojai\Documents\labelling\runs\tyre_seg3\weights\best.pt")
TEST_SOURCE = Path(r"C:\Users\lojai\Documents\labelling\yolo_whole_data_split\images\val")
SAVE_DIR    = Path(r"C:\Users\lojai\Documents\labelling\runs\tyre_seg_test")
CONF        = 0.5
IOU         = 0.45
# ──────────────────────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    sys.exit("ERROR: No CUDA GPU detected.")
if not MODEL_PATH.exists():
    sys.exit(f"ERROR: Model not found at {MODEL_PATH}")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))
print(f"Model  : {MODEL_PATH.parent.parent.name}")
print(f"Source : {TEST_SOURCE}")


def fit_circle(mask_xy):
    """Fit a minimum enclosing circle to mask contour points."""
    pts = np.array(mask_xy, dtype=np.float32)
    if len(pts) < 3:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(pts.reshape(-1, 1, 2).astype(np.float32))
    return int(cx), int(cy), int(r)


def draw_perfect_circles(img, boxes, masks, names):
    """Draw perfect circles fitted to each predicted mask."""
    overlay = img.copy()
    colors  = {0: (255, 100, 0), 1: (0, 120, 255)}   # outer=blue, inner=orange
    outline = {0: (255, 180, 0), 1: (0, 220, 0)}

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        xy   = mask.xy[0]

        circle = fit_circle(xy)
        if circle is None:
            continue
        cx, cy, r = circle

        cv2.circle(overlay, (cx, cy), r, colors[cls], -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.circle(img, (cx, cy), r, outline[cls], 3)

        label = f"{names[cls]} {conf:.2f}"
        cv2.putText(img, label, (cx - r, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, outline[cls], 2)

    return img


results = model.predict(
    source      = str(TEST_SOURCE),
    conf        = CONF,
    iou         = IOU,
    device      = 0,
    imgsz       = 640,
    save        = False,   # we save manually with perfect circles
    retina_masks = True,
    verbose     = False,
)

detected = 0
for r in results:
    img = cv2.imread(r.path)
    if img is None:
        continue

    if r.masks is not None and len(r.boxes):
        out = draw_perfect_circles(img, r.boxes, r.masks, model.names)
        detected += 1
    else:
        out = img

    out_path = SAVE_DIR / Path(r.path).name
    cv2.imwrite(str(out_path), out)

print(f"Detected in {detected}/{len(results)} images")
print(f"Saved to  : {SAVE_DIR}")
