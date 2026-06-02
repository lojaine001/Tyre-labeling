"""
STEP 0 (the_data) — Auto-annotate ALL images across all 7 folders (no deduplication).

Files that share the same tyre_id across folders are kept as separate entries,
prefixed with their folder number (e.g. f4_91137_top_91137-top.png).

Output:
  the_data_output/inner_and_outer/
    annotated_images/
    labels/
    classes.txt
    annotations.json
"""

import cv2
import numpy as np
import json
import math
from pathlib import Path

ROOT      = Path(r"c:\Users\lojai\Documents\labelling\the_data")
OUT_DIR   = Path(r"c:\Users\lojai\Documents\labelling\the_data_output\inner_and_outer")
POLY_POINTS = 72
CLASSES     = ["tyre_outer", "tyre_inner_hole"]

(OUT_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)


def detect_circles(img_bgr):
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    cx, cy  = w / 2, h / 2
    d       = min(w, h)

    def best(min_r, max_r, p2):
        c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                             param1=80, param2=p2,
                             minRadius=int(min_r), maxRadius=int(max_r))
        if c is None:
            return None
        return sorted(c[0], key=lambda x: abs(x[0] - cx) + abs(x[1] - cy))[0]

    outer = best(d * 0.30, d * 0.60, 50)
    inner = best(d * 0.15, d * 0.38, 60)
    return outer, inner


def circle_to_poly(cx, cy, r):
    angles = np.linspace(0, 2 * math.pi, POLY_POINTS, endpoint=False)
    return [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a))) for a in angles]


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


def poly_area(pts):
    n = len(pts)
    return abs(sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))) / 2


def poly_bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]


coco = {
    "info": {"description": "the_data — all 7 folders, all images including duplicates"},
    "categories": [
        {"id": 1, "name": "tyre_outer",      "supercategory": "tyre"},
        {"id": 2, "name": "tyre_inner_hole", "supercategory": "tyre"},
    ],
    "images":      [],
    "annotations": [],
}

ann_id  = 1
img_uid = 1   # unique incrementing image id (avoids collision across folders)
ok = 0
warn = 0

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    meta = json.loads((folder / "metadata.json").read_text())

    for img_meta in meta["images"]:
        tyre_id      = img_meta["tyre_id"]
        orig_pattern = f"{tyre_id}_top_{tyre_id}-top.png"
        img_path     = folder / orig_pattern

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w   = img.shape[:2]
        outer, inner = detect_circles(img)

        if outer is None or inner is None:
            print(f"  WARN: detection failed for {tyre_id} (folder {folder.name})")
            warn += 1
            continue

        ox, oy, or_ = float(outer[0]), float(outer[1]), float(outer[2])
        ix, iy, ir  = float(inner[0]), float(inner[1]), float(inner[2])

        outer_poly = circle_to_poly(ox, oy, or_)
        inner_poly = circle_to_poly(ix, iy, ir)

        # Unique filename: prefix with folder number
        stem    = f"f{folder.name}_{tyre_id}_top_{tyre_id}-top"
        pattern = f"{stem}.png"

        img_entry = {
            "id":         img_uid,
            "tyre_id":    tyre_id,
            "file_name":  pattern,
            "folder":     folder.name,
            "width":      w,
            "height":     h,
            "tyre_brand": img_meta.get("tyre_brand", ""),
            "tyre_size":  img_meta.get("tyre_size",  ""),
            "dot_code":   img_meta.get("dot_code",   ""),
        }
        coco["images"].append(img_entry)

        for cat_id, poly in [(1, outer_poly), (2, inner_poly)]:
            flat = [coord for pt in poly for coord in pt]
            coco["annotations"].append({
                "id":           ann_id,
                "image_id":     img_uid,
                "category_id":  cat_id,
                "segmentation": [flat],
                "area":         float(poly_area(poly)),
                "bbox":         poly_bbox(poly),
                "iscrowd":      0,
            })
            ann_id += 1

        def norm(pts):
            return " ".join(f"{round(x/w,6)} {round(y/h,6)}" for x, y in pts)
        (OUT_DIR / "labels" / f"{stem}.txt").write_text(
            f"0 {norm(outer_poly)}\n1 {norm(inner_poly)}\n"
        )

        ann_img = draw_annotated(img, ox, oy, or_, ix, iy, ir)
        cv2.imwrite(str(OUT_DIR / "annotated_images" / f"{stem}_annotated.png"), ann_img)

        img_uid += 1
        ok += 1

(OUT_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n")
with open(OUT_DIR / "annotations.json", "w") as f:
    json.dump(coco, f, indent=2)

print(f"\nDone: {ok} annotated, {warn} warnings.")
print(f"Output -> {OUT_DIR}")