"""
STEP 4 — Rebuild all annotated images and YOLO labels from authoritative sources.

For corrected images (in correct_progress.json): uses exact saved cx,cy,r values.
For uncorrected images: re-runs Hough detection (same params as original).

Updates:
  data_with_output/final/annotated_images/   <- all 154 annotated PNGs
  data_with_output/final/labels/             <- all 154 YOLO .txt files
  yolo_dataset/annotated_images/             <- same PNGs copied here
  yolo_dataset/labels/train|val|test/        <- split labels updated
"""

import cv2
import numpy as np
import json
import math
import shutil
from pathlib import Path

DATA_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_with")
FINAL_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_with_output\final")
AUTO_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_with_output\inner_and_outer")
YOLO_DIR   = Path(r"c:\Users\lojai\Documents\labelling\yolo_dataset")
CORR_FILE  = Path(r"c:\Users\lojai\Documents\labelling\correct_progress.json")
ANN_JSON   = AUTO_DIR / "annotations.json"   # source of truth for image list

POLY_POINTS = 72

# Hough params (same as annotate_data_with.py v1)
OUTER_MIN_R, OUTER_MAX_R, OUTER_P2 = 480, 780, 50
INNER_MIN_R, INNER_MAX_R, INNER_P2 = 250, 480, 60

corrected = {}
if CORR_FILE.exists():
    corrected = json.loads(CORR_FILE.read_text())

with open(ANN_JSON) as f:
    final_coco = json.load(f)

(FINAL_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "labels").mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)

# Build lookup: stem -> which yolo labels split folder
yolo_label_dirs = [YOLO_DIR / "labels" / s for s in ("train", "val", "test")]


def circle_to_poly(cx, cy, r):
    angles = np.linspace(0, 2 * math.pi, POLY_POINTS, endpoint=False)
    return np.array([(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles], dtype=np.float32)


def draw_annotated(img, ox, oy, or_, ix, iy, ir):
    overlay = img.copy()
    outer_pts = circle_to_poly(ox, oy, or_).astype(np.int32)
    inner_pts = circle_to_poly(ix, iy, ir).astype(np.int32)
    cv2.fillPoly(overlay, [outer_pts], (255, 100, 0))
    cv2.fillPoly(overlay, [inner_pts], (0, 120, 255))
    canvas = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(canvas, [outer_pts], True, (255, 180, 0), 3)
    cv2.polylines(canvas, [inner_pts], True, (0, 220, 0), 3)
    return canvas


def hough_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    cx_img, cy_img = w / 2, h / 2

    def best(min_r, max_r, p2):
        c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                             param1=80, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is None:
            return None
        return sorted(c[0], key=lambda x: abs(x[0] - cx_img) + abs(x[1] - cy_img))[0]

    outer = best(OUTER_MIN_R, OUTER_MAX_R, OUTER_P2)
    inner = best(INNER_MIN_R, INNER_MAX_R, INNER_P2)
    return outer, inner


def make_yolo_label(img_w, img_h, ox, oy, or_, ix, iy, ir):
    outer_pts = circle_to_poly(ox, oy, or_)
    inner_pts = circle_to_poly(ix, iy, ir)

    def norm(pts):
        return " ".join(
            f"{round(x / img_w, 6)} {round(y / img_h, 6)}"
            for x, y in pts
        )

    return f"0 {norm(outer_pts)}\n1 {norm(inner_pts)}\n"


ok = 0
used_corrected = 0
skipped = 0

for img_info in final_coco["images"]:
    tid     = str(img_info["id"])
    pattern = img_info["file_name"]         # e.g. "90357_top_90357-top.png"
    stem    = pattern.replace(".png", "")   # e.g. "90357_top_90357-top"
    ann_png = f"{stem}_annotated.png"

    img_path = DATA_DIR / pattern
    if not img_path.exists():
        print(f"  SKIP (not found): {img_path}")
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  SKIP (unreadable): {img_path}")
        skipped += 1
        continue

    h, w = img.shape[:2]

    if tid in corrected:
        c = corrected[tid]
        ox, oy, or_ = c["outer"]
        ix, iy, ir  = c["inner"]
        used_corrected += 1
    else:
        outer, inner = hough_detect(img)
        if outer is None or inner is None:
            print(f"  WARN: circle detection failed for {tid}")
            skipped += 1
            continue
        ox, oy, or_ = float(outer[0]), float(outer[1]), float(outer[2])
        ix, iy, ir  = float(inner[0]), float(inner[1]), float(inner[2])

    # Generate annotated image
    ann_img = draw_annotated(img, ox, oy, or_, ix, iy, ir)
    cv2.imwrite(str(FINAL_DIR / "annotated_images" / ann_png), ann_img)
    cv2.imwrite(str(YOLO_DIR  / "annotated_images" / ann_png), ann_img)

    # Generate YOLO label
    label_text = make_yolo_label(w, h, ox, oy, or_, ix, iy, ir)
    label_name = f"{stem}.txt"

    (FINAL_DIR / "labels" / label_name).write_text(label_text)

    # Update whichever split folder contains this label
    for split_dir in yolo_label_dirs:
        dst = split_dir / label_name
        if dst.exists():
            dst.write_text(label_text)
            break

    ok += 1

# Copy annotations.json and classes.txt from auto dir to final
import shutil
shutil.copy2(AUTO_DIR / "annotations.json", FINAL_DIR / "annotations.json")
shutil.copy2(AUTO_DIR / "classes.txt",      FINAL_DIR / "classes.txt")

auto = ok - used_corrected
print(f"\nRebuilt {ok} images:")
print(f"  From correct_progress.json : {used_corrected}")
print(f"  From Hough auto-detection  : {auto}")
if skipped:
    print(f"  Skipped                    : {skipped}")
print(f"\nUpdated:")
print(f"  {FINAL_DIR / 'annotated_images'}")
print(f"  {FINAL_DIR / 'labels'}")
print(f"  {YOLO_DIR  / 'annotated_images'}")
print(f"  yolo_dataset/labels/train|val|test/ (where applicable)")
