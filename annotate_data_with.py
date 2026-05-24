"""
Annotate data_with/ tyres (images with background).

Detection: Hough circle transform for outer tyre boundary and inner hole.

Outputs (two separate dataset folders):
  data_with_output/
    inner_only/
      annotated_images/   PNGs with inner hole drawn
      labels/             YOLO .txt files (class 0 = tyre_inner_hole)
      classes.txt
      annotations.json    COCO JSON
    inner_and_outer/
      annotated_images/   PNGs with outer + inner drawn
      labels/             YOLO .txt files (class 0 = tyre_outer, 1 = tyre_inner_hole)
      classes.txt
      annotations.json    COCO JSON
"""

import json
import cv2
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"c:\Users\lojai\Documents\labelling\data_with")
OUT_ROOT = Path(r"c:\Users\lojai\Documents\labelling\data_with_output")

# Hough parameters
HOUGH_BLUR   = (11, 11)
HOUGH_SIGMA  = 3
HOUGH_P1     = 80       # Canny high threshold
OUTER_P2     = 50       # accumulator threshold for outer circle
INNER_P2     = 60
OUTER_MIN_R  = 480
OUTER_MAX_R  = 780
INNER_MIN_R  = 250
INNER_MAX_R  = 480

POLY_POINTS  = 72       # polygon sample points around each circle

CLASSES_INNER_ONLY  = ["tyre_inner_hole"]
CLASSES_BOTH        = ["tyre_outer", "tyre_inner_hole"]

VIS = {
    "tyre_outer":      {"fill": (255, 100,   0), "line": (255, 180,   0)},
    "tyre_inner_hole": {"fill": (  0, 120, 255), "line": (  0, 220,   0)},
}


def detect_circles(img_bgr):
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, HOUGH_BLUR, HOUGH_SIGMA)
    cx_img, cy_img = w / 2, h / 2

    def best_circle(min_r, max_r, p2):
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=100,
                                   param1=HOUGH_P1, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is None:
            return None
        return sorted(circles[0], key=lambda c: abs(c[0] - cx_img) + abs(c[1] - cy_img))[0]

    outer = best_circle(OUTER_MIN_R, OUTER_MAX_R, OUTER_P2)
    inner = best_circle(INNER_MIN_R, INNER_MAX_R, INNER_P2)
    return outer, inner   # each is (cx, cy, r) or None


def circle_to_polygon(cx, cy, r, n=POLY_POINTS):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = [(float(cx + r * np.cos(a)), float(cy + r * np.sin(a))) for a in angles]
    return pts


def polygon_to_yolo(pts, img_w, img_h):
    flat = []
    for x, y in pts:
        flat.append(round(x / img_w, 6))
        flat.append(round(y / img_h, 6))
    return flat


def polygon_area(pts):
    n = len(pts)
    area = 0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def polygon_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x, y = min(xs), min(ys)
    return [x, y, max(xs) - x, max(ys) - y]


def draw_circle_on_canvas(canvas, overlay, cx, cy, r, class_name):
    pts = np.array(circle_to_polygon(cx, cy, r), dtype=np.int32)
    cv2.fillPoly(overlay, [pts], VIS[class_name]["fill"])
    return overlay


def draw_outlines(canvas, circles_info):
    for cx, cy, r, class_name in circles_info:
        pts = np.array(circle_to_polygon(cx, cy, r), dtype=np.int32)
        cv2.polylines(canvas, [pts], True, VIS[class_name]["line"], 4)


def empty_coco(description, metadata, categories):
    return {
        "info": {
            "description": description,
            "dataset":     metadata.get("dataset_name", ""),
            "created_at":  metadata.get("created_at", ""),
        },
        "categories": [{"id": i + 1, "name": n, "supercategory": "tyre"}
                       for i, n in enumerate(categories)],
        "images":      [],
        "annotations": [],
    }


def setup_dirs(variant):
    d = OUT_ROOT / variant
    (d / "annotated_images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    return d


def main():
    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    dir_inner = setup_dirs("inner_only")
    dir_both  = setup_dirs("inner_and_outer")

    coco_inner = empty_coco("Tyre inner hole segmentation",         metadata, CLASSES_INNER_ONLY)
    coco_both  = empty_coco("Tyre outer + inner hole segmentation", metadata, CLASSES_BOTH)

    # Map class name -> COCO category id
    cat_inner_only = {"tyre_inner_hole": 1}
    cat_both       = {"tyre_outer": 1, "tyre_inner_hole": 2}
    # YOLO class ids (0-indexed)
    yolo_inner_only = {"tyre_inner_hole": 0}
    yolo_both       = {"tyre_outer": 0, "tyre_inner_hole": 1}

    ann_id_inner = 1
    ann_id_both  = 1
    ok = 0
    warn = 0

    for img_meta in metadata["images"]:
        tyre_id = img_meta["tyre_id"]
        pattern = f"{tyre_id}_top_{tyre_id}-top.png"
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
            print(f"  WARN: {'outer' if outer is None else 'inner'} not found for {tyre_id}")
            warn += 1
            continue

        ox, oy, or_ = float(outer[0]), float(outer[1]), float(outer[2])
        ix, iy, ir  = float(inner[0]), float(inner[1]), float(inner[2])

        outer_poly = circle_to_polygon(ox, oy, or_)
        inner_poly = circle_to_polygon(ix, iy, ir)

        img_entry = {
            "id":         tyre_id,
            "file_name":  pattern,
            "width":      w,
            "height":     h,
            "tyre_brand": img_meta.get("tyre_brand", ""),
            "tyre_size":  img_meta.get("tyre_size", ""),
            "dot_code":   img_meta.get("dot_code", ""),
        }

        # ---- inner-only ----
        coco_inner["images"].append(img_entry)

        flat_inner = polygon_to_yolo(inner_poly, w, h)
        area_inner = polygon_area(inner_poly)
        bbox_inner = polygon_bbox(inner_poly)

        coco_inner["annotations"].append({
            "id":          ann_id_inner,
            "image_id":    tyre_id,
            "category_id": cat_inner_only["tyre_inner_hole"],
            "segmentation": [[coord for pt in inner_poly for coord in pt]],
            "area":        float(area_inner),
            "bbox":        bbox_inner,
            "iscrowd":     0,
        })
        ann_id_inner += 1

        yolo_line = f"{yolo_inner_only['tyre_inner_hole']} " + " ".join(map(str, flat_inner))
        (dir_inner / "labels" / pattern.replace(".png", ".txt")).write_text(yolo_line + "\n")

        overlay = img.copy()
        overlay = draw_circle_on_canvas(img, overlay, ix, iy, ir, "tyre_inner_hole")
        vis_inner = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        draw_outlines(vis_inner, [(ix, iy, ir, "tyre_inner_hole")])
        cv2.imwrite(str(dir_inner / "annotated_images" / pattern.replace(".png", "_annotated.png")), vis_inner)

        # ---- inner + outer ----
        coco_both["images"].append(img_entry)

        flat_outer = polygon_to_yolo(outer_poly, w, h)
        area_outer = polygon_area(outer_poly)
        bbox_outer = polygon_bbox(outer_poly)

        coco_both["annotations"].append({
            "id":          ann_id_both,
            "image_id":    tyre_id,
            "category_id": cat_both["tyre_outer"],
            "segmentation": [[coord for pt in outer_poly for coord in pt]],
            "area":        float(area_outer),
            "bbox":        bbox_outer,
            "iscrowd":     0,
        })
        ann_id_both += 1
        coco_both["annotations"].append({
            "id":          ann_id_both,
            "image_id":    tyre_id,
            "category_id": cat_both["tyre_inner_hole"],
            "segmentation": [[coord for pt in inner_poly for coord in pt]],
            "area":        float(area_inner),
            "bbox":        bbox_inner,
            "iscrowd":     0,
        })
        ann_id_both += 1

        yolo_lines = (
            f"{yolo_both['tyre_outer']} "      + " ".join(map(str, flat_outer)) + "\n" +
            f"{yolo_both['tyre_inner_hole']} " + " ".join(map(str, flat_inner)) + "\n"
        )
        (dir_both / "labels" / pattern.replace(".png", ".txt")).write_text(yolo_lines)

        overlay = img.copy()
        overlay = draw_circle_on_canvas(img, overlay, ox, oy, or_, "tyre_outer")
        overlay = draw_circle_on_canvas(img, overlay, ix, iy, ir,  "tyre_inner_hole")
        vis_both = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        draw_outlines(vis_both, [(ox, oy, or_, "tyre_outer"), (ix, iy, ir, "tyre_inner_hole")])
        cv2.imwrite(str(dir_both / "annotated_images" / pattern.replace(".png", "_annotated.png")), vis_both)

        ok += 1

    # Write classes.txt
    (dir_inner / "classes.txt").write_text("\n".join(CLASSES_INNER_ONLY) + "\n")
    (dir_both  / "classes.txt").write_text("\n".join(CLASSES_BOTH)       + "\n")

    # Write COCO JSON
    with open(dir_inner / "annotations.json", "w") as f:
        json.dump(coco_inner, f, indent=2)
    with open(dir_both / "annotations.json", "w") as f:
        json.dump(coco_both, f, indent=2)

    print(f"\nDone. {ok} images annotated, {warn} warnings.")
    print(f"Inner only : {len(coco_inner['annotations'])} annotations -> {dir_inner}")
    print(f"Inner+Outer: {len(coco_both['annotations'])} annotations -> {dir_both}")


if __name__ == "__main__":
    main()
