"""
STEP 3 (data_without) — Merge auto-annotated good images with manual corrections.

Reads:
  bad_images_without.txt          -> IDs you flagged
  data_without_output/corrected/  -> your corrections
  data_without_output/inner_and_outer/ -> auto annotations

Writes final clean dataset to:
  data_without_output/final/
    annotated_images/
    labels/
    classes.txt
    annotations.json
"""

import cv2
import numpy as np
import json
import math
import shutil
from pathlib import Path

BAD_FILE  = Path(r"c:\Users\lojai\Documents\labelling\bad_images_without.txt")
AUTO_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\inner_and_outer")
CORR_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\corrected")
FINAL_DIR = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\final")
DONE_FILE = Path(r"c:\Users\lojai\Documents\labelling\correct_progress_without.json")

(FINAL_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "labels").mkdir(parents=True, exist_ok=True)

bad_ids = set()
if BAD_FILE.exists():
    bad_ids = {l.strip() for l in BAD_FILE.read_text().splitlines() if l.strip()}

corrected = {}
if DONE_FILE.exists():
    corrected = json.loads(DONE_FILE.read_text())

missing = bad_ids - set(corrected.keys())
if missing:
    print(f"WARNING: {len(missing)} bad images not yet corrected: {missing}")
    print("Run 2_correct_without.py first, or they will be excluded.\n")

with open(AUTO_DIR / "annotations.json") as f:
    auto_coco = json.load(f)

POLY_POINTS = 72

def circle_to_poly(cx, cy, r):
    angles = [2 * math.pi * i / POLY_POINTS for i in range(POLY_POINTS)]
    return [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a))) for a in angles]

def poly_area(pts):
    n = len(pts)
    return abs(sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))) / 2

def poly_bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]

def draw_annotated(img, ox, oy, or_, ix, iy, ir):
    overlay = img.copy()
    outer_pts = np.array(circle_to_poly(ox, oy, or_), dtype=np.int32)
    inner_pts = np.array(circle_to_poly(ix, iy, ir),  dtype=np.int32)
    cv2.fillPoly(overlay, [outer_pts], (255, 100, 0))
    cv2.fillPoly(overlay, [inner_pts], (0, 120, 255))
    canvas = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(canvas, [outer_pts], True, (255, 180, 0), 3)
    cv2.polylines(canvas, [inner_pts], True, (0, 220, 0), 3)
    return canvas

auto_anns_by_img = {}
for ann in auto_coco["annotations"]:
    auto_anns_by_img.setdefault(ann["image_id"], []).append(ann)

final_coco = {
    "info":        auto_coco["info"],
    "categories":  auto_coco["categories"],
    "images":      [],
    "annotations": [],
}
final_coco["info"]["description"] = "Tyre outer + inner hole — background removed, final"

ann_id  = 1
skipped = 0
n_corr  = 0

DATA_WITHOUT = Path(r"c:\Users\lojai\Documents\labelling\data_without")

for img_info in auto_coco["images"]:
    tid    = str(img_info["id"])
    pattern = img_info["file_name"]
    stem    = pattern.replace(".png", "")
    ann_png = f"{stem}_annotated.png"

    if tid in bad_ids:
        if tid not in corrected:
            skipped += 1
            continue

        c = corrected[tid]
        ox, oy, or_ = c["outer"]
        ix, iy, ir  = c["inner"]
        w, h = img_info["width"], img_info["height"]

        outer_poly = circle_to_poly(ox, oy, or_)
        inner_poly = circle_to_poly(ix, iy, ir)

        final_coco["images"].append(img_info)
        for cat_id, poly in [(1, outer_poly), (2, inner_poly)]:
            flat = [coord for pt in poly for coord in pt]
            final_coco["annotations"].append({
                "id":           ann_id,
                "image_id":     img_info["id"],
                "category_id":  cat_id,
                "segmentation": [flat],
                "area":         float(poly_area(poly)),
                "bbox":         poly_bbox(poly),
                "iscrowd":      0,
            })
            ann_id += 1

        # Regenerate annotated image from corrected circles
        img_path = DATA_WITHOUT / pattern
        if img_path.exists():
            img = cv2.imread(str(img_path))
            ann_img = draw_annotated(img, ox, oy, or_, ix, iy, ir)
            cv2.imwrite(str(FINAL_DIR / "annotated_images" / ann_png), ann_img)

        src_lbl = CORR_DIR / "labels" / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, FINAL_DIR / "labels" / f"{stem}.txt")

        n_corr += 1

    else:
        final_coco["images"].append(img_info)
        for ann in auto_anns_by_img.get(img_info["id"], []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            final_coco["annotations"].append(new_ann)
            ann_id += 1

        src = AUTO_DIR / "annotated_images" / ann_png
        dst = FINAL_DIR / "annotated_images" / ann_png
        if src.exists():
            shutil.copy2(src, dst)

        src_lbl = AUTO_DIR / "labels" / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, FINAL_DIR / "labels" / f"{stem}.txt")

with open(FINAL_DIR / "annotations.json", "w") as f:
    json.dump(final_coco, f, indent=2)
shutil.copy2(AUTO_DIR / "classes.txt", FINAL_DIR / "classes.txt")

total = len(final_coco["images"])
n_auto = total - n_corr
print(f"Final dataset (data_without): {total} images")
print(f"  Auto-annotated : {n_auto}")
print(f"  Manually fixed : {n_corr}")
if skipped:
    print(f"  Skipped        : {skipped}")
print(f"\nOutput -> {FINAL_DIR}")