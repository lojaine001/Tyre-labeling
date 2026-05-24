"""
STEP 2 — Manually correct circles for images flagged as bad.

Loads each bad image and lets you adjust the outer and inner circles
using sliders. When happy, press S to save and move to the next image.

Controls:
  S        -> save current circles and go to next image
  R        -> reset to auto-detected values
  LEFT     -> go back to previous image
  Q / ESC  -> quit (progress is saved)

Corrected annotations are saved to:
  data_with_output/corrected/labels/   (YOLO .txt)
  data_with_output/corrected/annotations.json  (COCO)
  data_with_output/corrected/annotated_images/ (preview PNGs)
"""

import cv2
import numpy as np
import os
import json
import math
from pathlib import Path

DATA_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_with")
BAD_FILE  = Path(r"c:\Users\lojai\Documents\labelling\bad_images.txt")
OUT_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_with_output\corrected")
DONE_FILE = Path(r"c:\Users\lojai\Documents\labelling\correct_progress.json")

(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)

POLY_POINTS = 72

if not BAD_FILE.exists():
    print("bad_images.txt not found. Run 1_review.py first.")
    exit()

bad_ids = [line.strip() for line in BAD_FILE.read_text().splitlines() if line.strip()]
if not bad_ids:
    print("No bad images to correct.")
    exit()

# Load existing corrected data
corrected = {}
if DONE_FILE.exists():
    corrected = json.loads(DONE_FILE.read_text())

def detect_auto(img_bgr):
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (11, 11), 3)
    cx_img, cy_img = w / 2, h / 2
    def best(min_r, max_r, p2):
        c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                             param1=80, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is None:
            return None
        return sorted(c[0], key=lambda x: abs(x[0] - cx_img) + abs(x[1] - cy_img))[0]
    outer = best(480, 780, 50)
    inner = best(250, 480, 60)
    return outer, inner

def circle_to_poly(cx, cy, r, n=POLY_POINTS):
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a))) for a in angles]

def save_annotation(tid, img_w, img_h, outer, inner):
    ox, oy, or_ = outer
    ix, iy, ir  = inner

    outer_poly = circle_to_poly(ox, oy, or_)
    inner_poly = circle_to_poly(ix, iy, ir)

    def norm(pts):
        flat = []
        for x, y in pts:
            flat += [round(x / img_w, 6), round(y / img_h, 6)]
        return flat

    lines = (
        "0 " + " ".join(map(str, norm(outer_poly))) + "\n" +
        "1 " + " ".join(map(str, norm(inner_poly))) + "\n"
    )
    (OUT_DIR / "labels" / f"{tid}_top_{tid}-top.txt").write_text(lines)
    corrected[tid] = {"outer": list(outer), "inner": list(inner)}
    DONE_FILE.write_text(json.dumps(corrected, indent=2))

def draw_preview(img, ox, oy, or_, ix, iy, ir_):
    overlay = img.copy()
    outer_pts = np.array(circle_to_poly(ox, oy, or_), dtype=np.int32)
    inner_pts = np.array(circle_to_poly(ix, iy, ir_), dtype=np.int32)
    cv2.fillPoly(overlay, [outer_pts], (255, 100, 0))
    cv2.fillPoly(overlay, [inner_pts], (0, 120, 255))
    canvas = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(canvas, [outer_pts], True, (255, 180, 0), 3)
    cv2.polylines(canvas, [inner_pts], True, (0, 220, 0), 3)
    return canvas

WIN = "Correct — S=Save  R=Reset  LEFT=Back  Q=Quit"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 900, 1000)

# Trackbar state
state = {}

def nothing(_): pass

def setup_trackbars(img_w, img_h, ox, oy, or_, ix, iy, ir_):
    for name in ["outer_x","outer_y","outer_r","inner_x","inner_y","inner_r"]:
        try: cv2.destroyWindow("__tb__")
        except: pass
    cv2.namedWindow("__tb__", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("__tb__", 500, 250)
    cv2.createTrackbar("outer_x", "__tb__", int(ox), img_w,  nothing)
    cv2.createTrackbar("outer_y", "__tb__", int(oy), img_h,  nothing)
    cv2.createTrackbar("outer_r", "__tb__", int(or_), 900,   nothing)
    cv2.createTrackbar("inner_x", "__tb__", int(ix), img_w,  nothing)
    cv2.createTrackbar("inner_y", "__tb__", int(iy), img_h,  nothing)
    cv2.createTrackbar("inner_r", "__tb__", int(ir_), 600,   nothing)

def get_trackbars():
    return (
        cv2.getTrackbarPos("outer_x", "__tb__"),
        cv2.getTrackbarPos("outer_y", "__tb__"),
        cv2.getTrackbarPos("outer_r", "__tb__"),
        cv2.getTrackbarPos("inner_x", "__tb__"),
        cv2.getTrackbarPos("inner_y", "__tb__"),
        cv2.getTrackbarPos("inner_r", "__tb__"),
    )

idx = 0
while 0 <= idx < len(bad_ids):
    tid      = bad_ids[idx]
    img_path = DATA_DIR / f"{tid}_top_{tid}-top.png"
    if not img_path.exists():
        print(f"  SKIP: {img_path} not found")
        idx += 1
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Use previously corrected values if available, else auto-detect
    if tid in corrected:
        auto_outer = corrected[tid]["outer"]
        auto_inner = corrected[tid]["inner"]
    else:
        ao, ai = detect_auto(img)
        auto_outer = list(ao) if ao is not None else [w//2, h//2, 600]
        auto_inner = list(ai) if ai is not None else [w//2, h//2, 380]

    setup_trackbars(w, h,
                    auto_outer[0], auto_outer[1], auto_outer[2],
                    auto_inner[0], auto_inner[1], auto_inner[2])

    print(f"\n[{idx+1}/{len(bad_ids)}] Correcting tyre {tid}  (S=save, R=reset, LEFT=back, Q=quit)")

    while True:
        ox, oy, or_, ix, iy, ir_ = get_trackbars()
        preview = draw_preview(img, ox, oy, or_, ix, iy, ir_)

        info = f"[{idx+1}/{len(bad_ids)}] {tid}  |  outer r={or_}  inner r={ir_}"
        hint = "S=Save   R=Reset   LEFT=Back   Q=Quit"
        cv2.rectangle(preview, (0, h-55), (w, h), (30,30,30), -1)
        cv2.putText(preview, info, (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(preview, hint, (10, h- 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(180,180,180), 1)

        cv2.imshow(WIN, preview)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            save_annotation(tid, w, h, (ox, oy, or_), (ix, iy, ir_))
            # Save annotated preview image
            ann_img = draw_preview(img, ox, oy, or_, ix, iy, ir_)
            cv2.imwrite(str(OUT_DIR / "annotated_images" / f"{tid}_top_{tid}-top_annotated.png"), ann_img)
            print(f"  Saved: outer=({ox},{oy},r{or_})  inner=({ix},{iy},r{ir_})")
            idx += 1
            break
        elif key == ord('r'):
            setup_trackbars(w, h,
                            auto_outer[0], auto_outer[1], auto_outer[2],
                            auto_inner[0], auto_inner[1], auto_inner[2])
        elif key == 81:   # LEFT
            idx = max(0, idx - 1)
            break
        elif key in (ord('q'), 27):
            idx = len(bad_ids)  # exit loop
            break

cv2.destroyAllWindows()
print(f"\nCorrected {len(corrected)} images so far. Run 3_merge.py when done.")
