"""
Generate two separate COCO JSON annotation files and two image folders:

  1. annotations_inner_only.json       + annotated_images_inner_only/
     -> tyre_inner_hole only

  2. annotations_inner_and_outer.json  + annotated_images_inner_and_outer/
     -> tyre_outer  +  tyre_inner_hole
"""

import json
import cv2
import numpy as np
from pathlib import Path

DATA_DIR     = Path(r"c:\Users\lojai\Documents\labelling\data")
OUT_INNER    = Path(r"c:\Users\lojai\Documents\labelling\annotations_inner_only.json")
OUT_BOTH     = Path(r"c:\Users\lojai\Documents\labelling\annotations_inner_and_outer.json")
DIR_INNER    = Path(r"c:\Users\lojai\Documents\labelling\annotated_images_inner_only")
DIR_BOTH     = Path(r"c:\Users\lojai\Documents\labelling\annotated_images_inner_and_outer")

CAT_OUTER = {"id": 1, "name": "tyre_outer",      "supercategory": "tyre"}
CAT_INNER = {"id": 2, "name": "tyre_inner_hole",  "supercategory": "tyre"}

POLY_EPSILON_FACTOR = 0.002

VIS = {
    1: {"fill": (255, 100,   0), "line": (255, 180,   0)},  # blue  – outer
    2: {"fill": (  0, 120, 255), "line": (  0, 220,   0)},  # orange – inner
}


def get_binary(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def find_contours(img_bgr):
    binary = get_binary(img_bgr)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return None, None
    hier = hierarchy[0]

    outer, outer_area = None, 0
    inner, inner_area = None, 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if hier[i][3] < 0:          # top-level = outer boundary
            if area > outer_area:
                outer_area, outer = area, cnt
        else:                        # child = hole inside ring
            if area > inner_area:
                inner_area, inner = area, cnt

    return (outer, outer_area), (inner, inner_area)


def contour_to_polygon(cnt):
    eps = POLY_EPSILON_FACTOR * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    return approx.flatten().tolist()


def make_annotation(ann_id, img_id, cat_id, cnt, area):
    polygon = contour_to_polygon(cnt)
    if len(polygon) < 6:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": cat_id,
        "segmentation": [polygon],
        "area": float(area),
        "bbox": [x, y, bw, bh],
        "iscrowd": 0,
    }


def draw_annotations(img, annotations_for_image):
    overlay = img.copy()
    canvas  = img.copy()
    for ann in annotations_for_image:
        pts = np.array(ann["segmentation"][0], dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(overlay, [pts], VIS[ann["category_id"]]["fill"])
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    for ann in annotations_for_image:
        pts = np.array(ann["segmentation"][0], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(canvas, [pts], True, VIS[ann["category_id"]]["line"], 4)
    return canvas


def empty_coco(description, metadata, categories):
    return {
        "info": {
            "description": description,
            "dataset": metadata.get("dataset_name", ""),
            "created_at": metadata.get("created_at", ""),
        },
        "categories": categories,
        "images": [],
        "annotations": [],
    }


def main():
    DIR_INNER.mkdir(exist_ok=True)
    DIR_BOTH.mkdir(exist_ok=True)

    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    coco_inner = empty_coco("Tyre inner hole segmentation",         metadata, [CAT_INNER])
    coco_both  = empty_coco("Tyre outer + inner hole segmentation", metadata, [CAT_OUTER, CAT_INNER])

    ann_id_inner = 1
    ann_id_both  = 1

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
        img_entry = {
            "id": tyre_id,
            "file_name": pattern,
            "width": w,
            "height": h,
            "tyre_brand": img_meta.get("tyre_brand", ""),
            "tyre_size":  img_meta.get("tyre_size", ""),
            "dot_code":   img_meta.get("dot_code", ""),
        }

        (outer_cnt, outer_area), (inner_cnt, inner_area) = find_contours(img)

        # --- inner-only ---
        inner_ann = make_annotation(ann_id_inner, tyre_id, CAT_INNER["id"], inner_cnt, inner_area) if inner_cnt is not None else None
        if inner_ann:
            coco_inner["images"].append(img_entry)
            coco_inner["annotations"].append(inner_ann)
            ann_id_inner += 1
            vis_inner = draw_annotations(img, [inner_ann])
            cv2.imwrite(str(DIR_INNER / pattern.replace(".png", "_annotated.png")), vis_inner)

        # --- inner + outer ---
        anns_both = []
        if outer_cnt is not None:
            a = make_annotation(ann_id_both, tyre_id, CAT_OUTER["id"], outer_cnt, outer_area)
            if a:
                anns_both.append(a)
                ann_id_both += 1
        if inner_cnt is not None:
            a = make_annotation(ann_id_both, tyre_id, CAT_INNER["id"], inner_cnt, inner_area)
            if a:
                anns_both.append(a)
                ann_id_both += 1

        if anns_both:
            coco_both["images"].append(img_entry)
            coco_both["annotations"].extend(anns_both)
            vis_both = draw_annotations(img, anns_both)
            cv2.imwrite(str(DIR_BOTH / pattern.replace(".png", "_annotated.png")), vis_both)

        print(f"  OK tyre {tyre_id}")

    with open(OUT_INNER, "w") as f:
        json.dump(coco_inner, f, indent=2)
    with open(OUT_BOTH, "w") as f:
        json.dump(coco_both, f, indent=2)

    print(f"\nInner only : {len(coco_inner['annotations'])} annotations -> {OUT_INNER.name}")
    print(f"Inner+Outer: {len(coco_both['annotations'])} annotations -> {OUT_BOTH.name}")


if __name__ == "__main__":
    main()
