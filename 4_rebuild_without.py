"""
STEP 4 (data_without) — Rebuild all annotated images and labels from authoritative sources,
then create yolo_dataset_without/ with train/val/test split.

For corrected images (correct_progress_without.json): uses exact saved circles.
For the 13 non-corrected images: re-runs contour detection.

Outputs:
  data_without_output/final/annotated_images/   <- rebuilt PNGs
  data_without_output/final/labels/             <- rebuilt YOLO .txt files
  yolo_dataset_without/
    images/train|val|test/
    labels/train|val|test/
    annotated_images/
    dataset.yaml
"""

import cv2
import numpy as np
import json
import math
import shutil
import random
from pathlib import Path

DATA_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_without")
FINAL_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\final")
YOLO_DIR   = Path(r"c:\Users\lojai\Documents\labelling\yolo_dataset_without")
CORR_FILE  = Path(r"c:\Users\lojai\Documents\labelling\correct_progress_without.json")
ANN_JSON   = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\inner_and_outer\annotations.json")

POLY_POINTS = 72
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
random.seed(42)

corrected = {}
if CORR_FILE.exists():
    corrected = json.loads(CORR_FILE.read_text())

with open(ANN_JSON) as f:
    coco = json.load(f)

for split in ("train", "val", "test"):
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "labels").mkdir(parents=True, exist_ok=True)


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


def contour_detect(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return None, None
    hier = hierarchy[0]
    outer_cnt, outer_area = None, 0
    inner_cnt, inner_area = None, 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if hier[i][3] < 0:
            if area > outer_area:
                outer_area, outer_cnt = area, cnt
        else:
            if area > inner_area:
                inner_area, inner_cnt = area, cnt
    if outer_cnt is None or inner_cnt is None:
        return None, None
    (ox, oy), or_ = cv2.minEnclosingCircle(outer_cnt)
    (ix, iy), ir  = cv2.minEnclosingCircle(inner_cnt)
    return (float(ox), float(oy), float(or_)), (float(ix), float(iy), float(ir))


def make_label(img_w, img_h, ox, oy, or_, ix, iy, ir):
    def norm(pts):
        return " ".join(f"{round(x/img_w,6)} {round(y/img_h,6)}" for x, y in pts)
    outer_pts = circle_to_poly(ox, oy, or_)
    inner_pts = circle_to_poly(ix, iy, ir)
    return f"0 {norm(outer_pts)}\n1 {norm(inner_pts)}\n"


# --- Rebuild all images ---
ok = 0
used_corrected = 0
skipped = 0
entries = []   # (stem, pattern) for split

for img_info in coco["images"]:
    tid     = str(img_info["id"])
    pattern = img_info["file_name"]
    stem    = pattern.replace(".png", "")
    ann_png = f"{stem}_annotated.png"

    img_path = DATA_DIR / pattern
    if not img_path.exists():
        print(f"  SKIP (not found): {pattern}")
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue

    h, w = img.shape[:2]

    if tid in corrected:
        c = corrected[tid]
        ox, oy, or_ = c["outer"]
        ix, iy, ir  = c["inner"]
        used_corrected += 1
    else:
        outer, inner = contour_detect(img)
        if outer is None or inner is None:
            print(f"  WARN: detection failed for {tid}")
            skipped += 1
            continue
        ox, oy, or_ = outer
        ix, iy, ir  = inner

    ann_img    = draw_annotated(img, ox, oy, or_, ix, iy, ir)
    label_text = make_label(w, h, ox, oy, or_, ix, iy, ir)
    label_name = f"{stem}.txt"

    cv2.imwrite(str(FINAL_DIR / "annotated_images" / ann_png), ann_img)
    (FINAL_DIR / "labels" / label_name).write_text(label_text)

    entries.append((stem, pattern, img_path, label_name, ann_png))
    ok += 1

# Copy classes.txt and annotations.json to final
shutil.copy2(
    Path(r"c:\Users\lojai\Documents\labelling\data_without_output\inner_and_outer\classes.txt"),
    FINAL_DIR / "classes.txt"
)
shutil.copy2(
    ANN_JSON,
    FINAL_DIR / "annotations.json"
)

# --- Create train/val/test split ---
random.shuffle(entries)
n = len(entries)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

splits = {
    "train": entries[:n_train],
    "val":   entries[n_train:n_train + n_val],
    "test":  entries[n_train + n_val:],
}

for split_name, split_entries in splits.items():
    for stem, pattern, img_src, label_name, ann_png in split_entries:
        shutil.copy2(img_src,
                     YOLO_DIR / "images" / split_name / pattern)
        shutil.copy2(FINAL_DIR / "labels" / label_name,
                     YOLO_DIR / "labels" / split_name / label_name)

    ann_dir = FINAL_DIR / "annotated_images"
    for _, _, _, _, ann_png in split_entries:
        src = ann_dir / ann_png
        if src.exists():
            shutil.copy2(src, YOLO_DIR / "annotated_images" / ann_png)

yaml_content = f"""# data_without YOLO dataset — background removed tyres
# Total: {ok} images

path: {YOLO_DIR.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: tyre_outer
  1: tyre_inner_hole
"""
(YOLO_DIR / "dataset.yaml").write_text(yaml_content)

auto = ok - used_corrected
print(f"Rebuilt {ok} images:")
print(f"  From correct_progress_without.json : {used_corrected}")
print(f"  From contour auto-detection        : {auto}")
if skipped:
    print(f"  Skipped                            : {skipped}")
print(f"\nyolo_dataset_without/:")
print(f"  Train : {len(splits['train'])}")
print(f"  Val   : {len(splits['val'])}")
print(f"  Test  : {len(splits['test'])}")
print(f"\nOutput -> {FINAL_DIR}")
print(f"          {YOLO_DIR}")