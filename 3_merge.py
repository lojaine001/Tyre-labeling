"""
STEP 3 — Merge auto-annotated good images with manually corrected ones.

Reads:
  bad_images.txt                        -> IDs you flagged
  data_with_output/corrected/           -> your manual corrections
  data_with_output/inner_and_outer/     -> the original auto annotations

Writes a final clean dataset to:
  data_with_output/final/
    annotated_images/
    labels/
    classes.txt
    annotations.json
"""

import json
import shutil
from pathlib import Path

BAD_FILE   = Path(r"c:\Users\lojai\Documents\labelling\bad_images.txt")
AUTO_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_with_output\inner_and_outer")
CORR_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_with_output\corrected")
FINAL_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_with_output\final")

(FINAL_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "labels").mkdir(parents=True, exist_ok=True)

bad_ids = set()
if BAD_FILE.exists():
    bad_ids = {l.strip() for l in BAD_FILE.read_text().splitlines() if l.strip()}

with open(AUTO_DIR / "annotations.json") as f:
    auto_coco = json.load(f)

# Build corrected circle lookup
corr_progress = Path(r"c:\Users\lojai\Documents\labelling\correct_progress.json")
corrected = {}
if corr_progress.exists():
    corrected = json.loads(corr_progress.read_text())

corrected_ids = set(corrected.keys())
missing_corrections = bad_ids - corrected_ids
if missing_corrections:
    print(f"WARNING: {len(missing_corrections)} bad images not yet corrected: {missing_corrections}")
    print("Run 2_correct.py to fix them first, or they will be excluded from final output.\n")

# --- Build final COCO JSON ---
final_coco = {
    "info":        auto_coco["info"],
    "categories":  auto_coco["categories"],
    "images":      [],
    "annotations": [],
}
final_coco["info"]["description"] = "Tyre outer + inner hole — final (auto + manual corrections)"

ann_id = 1
used_images = set()

# Helper: rebuild annotation from circle
import math
POLY_POINTS = 72

def circle_to_poly(cx, cy, r):
    angles = [2 * math.pi * i / POLY_POINTS for i in range(POLY_POINTS)]
    return [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a))) for a in angles]

def poly_area(pts):
    n = len(pts)
    area = sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))
    return abs(area) / 2

def poly_bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]

# Index auto annotations by image_id
auto_anns_by_img = {}
for ann in auto_coco["annotations"]:
    auto_anns_by_img.setdefault(ann["image_id"], []).append(ann)
auto_imgs_by_id = {img["id"]: img for img in auto_coco["images"]}

skipped = 0

for img_info in auto_coco["images"]:
    tid     = str(img_info["id"])
    img_id  = img_info["id"]
    pattern = img_info["file_name"]
    stem    = pattern.replace(".png", "")
    ann_png = f"{stem}_annotated.png"

    if tid in bad_ids:
        if tid not in corrected_ids:
            skipped += 1
            continue  # skip uncorrected bad images

        # Use corrected circles
        circ = corrected[tid]
        ox, oy, or_ = circ["outer"]
        ix, iy, ir  = circ["inner"]
        w, h = img_info["width"], img_info["height"]

        outer_poly = circle_to_poly(ox, oy, or_)
        inner_poly = circle_to_poly(ix, iy, ir)

        final_coco["images"].append(img_info)
        for cat_id, poly in [(1, outer_poly), (2, inner_poly)]:
            flat = [coord for pt in poly for coord in pt]
            final_coco["annotations"].append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  cat_id,
                "segmentation": [flat],
                "area":         float(poly_area(poly)),
                "bbox":         poly_bbox(poly),
                "iscrowd":      0,
            })
            ann_id += 1

        # Copy corrected annotated image
        src = CORR_DIR / "annotated_images" / ann_png
        dst = FINAL_DIR / "annotated_images" / ann_png
        if src.exists():
            shutil.copy2(src, dst)

        # Copy corrected label
        lbl_src = CORR_DIR / "labels" / f"{stem}.txt"
        lbl_dst = FINAL_DIR / "labels"  / f"{stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)

    else:
        # Use auto annotations as-is
        final_coco["images"].append(img_info)
        for ann in auto_anns_by_img.get(img_id, []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            final_coco["annotations"].append(new_ann)
            ann_id += 1

        src = AUTO_DIR / "annotated_images" / ann_png
        dst = FINAL_DIR / "annotated_images" / ann_png
        if src.exists():
            shutil.copy2(src, dst)

        lbl_src = AUTO_DIR / "labels" / f"{stem}.txt"
        lbl_dst = FINAL_DIR / "labels"  / f"{stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)

# Write final outputs
with open(FINAL_DIR / "annotations.json", "w") as f:
    json.dump(final_coco, f, indent=2)

shutil.copy2(AUTO_DIR / "classes.txt", FINAL_DIR / "classes.txt")

total   = len(final_coco["images"])
n_corr  = len([t for t in bad_ids if t in corrected_ids])
n_auto  = total - n_corr

print(f"Final dataset: {total} images")
print(f"  Auto-annotated : {n_auto}")
print(f"  Manually fixed : {n_corr}")
if skipped:
    print(f"  Skipped (not yet corrected): {skipped}")
print(f"\nOutput -> {FINAL_DIR}")
