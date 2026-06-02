"""
STEP 4 (the_data) — Rebuild all annotated images and labels, then create
yolo_dataset_the_data/ as one flat dataset (no train/val/test split).

For corrected images: uses correct_progress_data.json.
For auto-annotated: re-runs Hough detection.
"""

import cv2, numpy as np, json, math, shutil, random
from pathlib import Path

ROOT       = Path(r"c:\Users\lojai\Documents\labelling\the_data")
FINAL_DIR  = Path(r"c:\Users\lojai\Documents\labelling\the_data_output\final")
YOLO_DIR   = Path(r"c:\Users\lojai\Documents\labelling\yolo_dataset_the_data")
CORR_FILE  = Path(r"c:\Users\lojai\Documents\labelling\correct_progress_data.json")
ANN_JSON   = Path(r"c:\Users\lojai\Documents\labelling\the_data_output\inner_and_outer\annotations.json")

POLY_POINTS = 72

corrected = {}
if CORR_FILE.exists():
    corrected = json.loads(CORR_FILE.read_text())

with open(ANN_JSON) as f:
    coco = json.load(f)

(YOLO_DIR / "images").mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "labels").mkdir(parents=True, exist_ok=True)
(YOLO_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "labels").mkdir(parents=True, exist_ok=True)


def circle_to_poly(cx, cy, r):
    a = np.linspace(0, 2*math.pi, POLY_POINTS, endpoint=False)
    return np.array([(cx + r*math.cos(x), cy + r*math.sin(x)) for x in a], dtype=np.float32)

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

def hough_detect(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    d = min(w, h)
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    cx, cy = w/2, h/2
    def best(mn, mx, p2):
        c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                             param1=80, param2=p2, minRadius=int(mn), maxRadius=int(mx))
        if c is None: return None
        return sorted(c[0], key=lambda x: abs(x[0]-cx)+abs(x[1]-cy))[0]
    outer = best(d*0.30, d*0.60, 50)
    inner = best(d*0.15, d*0.38, 60)
    return outer, inner

def make_label(img_w, img_h, ox, oy, or_, ix, iy, ir):
    def norm(pts):
        return " ".join(f"{round(x/img_w,6)} {round(y/img_h,6)}" for x, y in pts)
    return f"0 {norm(circle_to_poly(ox,oy,or_))}\n1 {norm(circle_to_poly(ix,iy,ir))}\n"


ok = 0
used_corrected = 0
skipped = 0

for img_info in coco["images"]:
    uid     = str(img_info["id"])
    tyre_id = str(img_info["tyre_id"])
    stem    = img_info["file_name"].replace(".png", "")   # e.g. f4_91137_top_91137-top
    ann_png = f"{stem}_annotated.png"
    folder  = img_info.get("folder", "4")

    orig_img_name = f"{tyre_id}_top_{tyre_id}-top.png"
    img_path = ROOT / folder / orig_img_name

    if not img_path.exists():
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue

    h, w = img.shape[:2]

    # Correction key is folder-prefixed stem so each folder's copy is independent
    corr_key = stem
    if corr_key in corrected:
        c = corrected[corr_key]
        ox, oy, or_ = c["outer"]
        ix, iy, ir  = c["inner"]
        used_corrected += 1
    else:
        outer, inner = hough_detect(img)
        if outer is None or inner is None:
            skipped += 1
            continue
        ox, oy, or_ = float(outer[0]), float(outer[1]), float(outer[2])
        ix, iy, ir  = float(inner[0]), float(inner[1]), float(inner[2])

    label_name = f"{stem}.txt"
    label_text = make_label(w, h, ox, oy, or_, ix, iy, ir)
    ann_img    = draw_annotated(img, ox, oy, or_, ix, iy, ir)

    # Save to final/
    (FINAL_DIR / "labels" / label_name).write_text(label_text)
    cv2.imwrite(str(FINAL_DIR / "annotated_images" / ann_png), ann_img)

    # Copy to flat yolo_dataset_the_data/ (image renamed to stem.png)
    yolo_img_name = f"{stem}.png"
    shutil.copy2(img_path, YOLO_DIR / "images" / yolo_img_name)
    (YOLO_DIR / "labels" / label_name).write_text(label_text)
    cv2.imwrite(str(YOLO_DIR / "annotated_images" / ann_png), ann_img)

    ok += 1

shutil.copy2(ANN_JSON, FINAL_DIR / "annotations.json")
shutil.copy2(
    Path(r"c:\Users\lojai\Documents\labelling\the_data_output\inner_and_outer\classes.txt"),
    FINAL_DIR / "classes.txt"
)

yaml_content = f"""# the_data YOLO dataset — all {ok} images, no split
# Use this as a single pool; split during training with val_split= parameter

path: {YOLO_DIR.as_posix()}
train: images
val:   images

nc: 2
names:
  0: tyre_outer
  1: tyre_inner_hole
"""
(YOLO_DIR / "dataset.yaml").write_text(yaml_content)

auto = ok - used_corrected
print(f"Rebuilt {ok} images:")
print(f"  From correct_progress_data.json : {used_corrected}")
print(f"  From Hough auto-detection       : {auto}")
if skipped: print(f"  Skipped                         : {skipped}")
print(f"\nyolo_dataset_the_data/: {ok} images (flat, no split)")
print(f"\nOutput -> {FINAL_DIR}")
print(f"          {YOLO_DIR}")