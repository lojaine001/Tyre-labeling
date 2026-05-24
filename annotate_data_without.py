"""
STEP 0 (data_without) — Auto-annotate background-removed tyre images.

Uses contour detection (works well since background is transparent/white).
Fits a minimum-enclosing circle to each contour so output format matches
data_with/ (72-point circle polygon, YOLO + COCO + annotated PNGs).

Outputs:
  data_without_output/inner_and_outer/
    annotated_images/   PNGs with outer + inner drawn
    labels/             YOLO .txt files (class 0 = tyre_outer, class 1 = tyre_inner_hole)
    classes.txt
    annotations.json    COCO JSON
"""

import cv2
import numpy as np
import json
import math
from pathlib import Path

DATA_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_without")
OUT_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\inner_and_outer")

POLY_POINTS = 72
CLASSES     = ["tyre_outer", "tyre_inner_hole"]

(OUT_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

VIS = {
    "outer": {"fill": (255, 100,   0), "line": (255, 180,   0)},
    "inner": {"fill": (  0, 120, 255), "line": (  0, 220,   0)},
}


def get_binary(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Background is white (255), tyre rubber is dark (50-150) -> invert so tyre = foreground
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def detect_circles(img_bgr):
    """Find outer tyre boundary and inner hole using contour hierarchy."""
    binary = get_binary(img_bgr)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return None, None
    hier = hierarchy[0]

    outer_cnt, outer_area = None, 0
    inner_cnt, inner_area = None, 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if hier[i][3] < 0:          # top-level contour = outer tyre boundary
            if area > outer_area:
                outer_area, outer_cnt = area, cnt
        else:                        # child contour = inner hole
            if area > inner_area:
                inner_area, inner_cnt = area, cnt

    if outer_cnt is None or inner_cnt is None:
        return None, None

    # Fit minimum enclosing circle to each contour
    (ox, oy), or_ = cv2.minEnclosingCircle(outer_cnt)
    (ix, iy), ir  = cv2.minEnclosingCircle(inner_cnt)
    return (float(ox), float(oy), float(or_)), (float(ix), float(iy), float(ir))


def circle_to_poly(cx, cy, r):
    angles = np.linspace(0, 2 * math.pi, POLY_POINTS, endpoint=False)
    return [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a))) for a in angles]


def draw_annotated(img, ox, oy, or_, ix, iy, ir):
    overlay = img.copy()
    outer_pts = np.array(circle_to_poly(ox, oy, or_), dtype=np.int32)
    inner_pts = np.array(circle_to_poly(ix, iy, ir),  dtype=np.int32)
    cv2.fillPoly(overlay, [outer_pts], VIS["outer"]["fill"])
    cv2.fillPoly(overlay, [inner_pts], VIS["inner"]["fill"])
    canvas = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(canvas, [outer_pts], True, VIS["outer"]["line"], 3)
    cv2.polylines(canvas, [inner_pts], True, VIS["inner"]["line"], 3)
    return canvas


def poly_area(pts):
    n = len(pts)
    return abs(sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))) / 2


def poly_bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]


with open(DATA_DIR / "metadata.json") as f:
    metadata = json.load(f)

coco = {
    "info": {
        "description": "Tyre outer + inner hole — background removed",
        "dataset":     metadata.get("dataset_name", ""),
        "created_at":  metadata.get("created_at", ""),
    },
    "categories": [
        {"id": 1, "name": "tyre_outer",      "supercategory": "tyre"},
        {"id": 2, "name": "tyre_inner_hole", "supercategory": "tyre"},
    ],
    "images":      [],
    "annotations": [],
}

ann_id = 1
ok = 0
warn = 0

for img_meta in metadata["images"]:
    tyre_id = img_meta["tyre_id"]
    pattern = f"{tyre_id}_top_removed_bg_{tyre_id}-top_removed_bg.png"
    img_path = DATA_DIR / pattern

    if not img_path.exists():
        print(f"  SKIP (not found): {pattern}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  SKIP (unreadable): {pattern}")
        continue

    h, w = img.shape[:2]
    outer, inner = detect_circles(img)

    if outer is None or inner is None:
        print(f"  WARN: circle detection failed for {tyre_id}")
        warn += 1
        continue

    ox, oy, or_ = outer
    ix, iy, ir  = inner

    outer_poly = circle_to_poly(ox, oy, or_)
    inner_poly = circle_to_poly(ix, iy, ir)

    img_entry = {
        "id":         tyre_id,
        "file_name":  pattern,
        "width":      w,
        "height":     h,
        "tyre_brand": img_meta.get("tyre_brand", ""),
        "tyre_size":  img_meta.get("tyre_size", ""),
        "dot_code":   img_meta.get("dot_code", ""),
    }
    coco["images"].append(img_entry)

    for cat_id, poly in [(1, outer_poly), (2, inner_poly)]:
        flat = [coord for pt in poly for coord in pt]
        coco["annotations"].append({
            "id":           ann_id,
            "image_id":     tyre_id,
            "category_id":  cat_id,
            "segmentation": [flat],
            "area":         float(poly_area(poly)),
            "bbox":         poly_bbox(poly),
            "iscrowd":      0,
        })
        ann_id += 1

    # YOLO label
    def norm(pts):
        return " ".join(f"{round(x/w, 6)} {round(y/h, 6)}" for x, y in pts)
    label = f"0 {norm(outer_poly)}\n1 {norm(inner_poly)}\n"
    stem = pattern.replace(".png", "")
    (OUT_DIR / "labels" / f"{stem}.txt").write_text(label)

    # Annotated image
    ann_img = draw_annotated(img, ox, oy, or_, ix, iy, ir)
    cv2.imwrite(str(OUT_DIR / "annotated_images" / f"{stem}_annotated.png"), ann_img)

    ok += 1

(OUT_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n")

with open(OUT_DIR / "annotations.json", "w") as f:
    json.dump(coco, f, indent=2)

print(f"\nDone. {ok} images annotated, {warn} warnings.")
print(f"Output -> {OUT_DIR}")