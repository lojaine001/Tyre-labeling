"""
STEP 2 (data_without) — Manually correct circles for bad background-removed images.

Uses circle sliders. Initial values come from fitting a circle to the detected contour.

Controls:
  S        -> save and go to next
  R        -> reset to auto-detected values
  LEFT     -> go back
  Q / ESC  -> quit
"""

import cv2
import numpy as np
import json
import math
from pathlib import Path

DATA_DIR  = Path(r"c:\Users\lojai\Documents\labelling\data_without")
BAD_FILE  = Path(r"c:\Users\lojai\Documents\labelling\bad_images_without.txt")
OUT_DIR   = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\corrected")
DONE_FILE = Path(r"c:\Users\lojai\Documents\labelling\correct_progress_without.json")

(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)

POLY_POINTS = 72

ANN_JSON = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\inner_and_outer\annotations.json")

# If bad_images_without.txt exists use it; otherwise correct ALL images
if BAD_FILE.exists():
    bad_ids = [l.strip() for l in BAD_FILE.read_text().splitlines() if l.strip()]
    if not bad_ids:
        print("No bad images flagged — running over all images instead.")
        with open(ANN_JSON) as f:
            d = json.load(f)
        bad_ids = [str(img["id"]) for img in d["images"]]
else:
    print("No bad_images_without.txt found — running over all images.")
    with open(ANN_JSON) as f:
        d = json.load(f)
    bad_ids = [str(img["id"]) for img in d["images"]]

corrected = {}
if DONE_FILE.exists():
    corrected = json.loads(DONE_FILE.read_text())


def get_binary(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def detect_auto(img_bgr):
    binary = get_binary(img_bgr)
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
    return [float(ox), float(oy), float(or_)], [float(ix), float(iy), float(ir)]


def circle_to_poly(cx, cy, r, n=POLY_POINTS):
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a))) for a in angles]


def save_annotation(tid, img_w, img_h, outer, inner):
    ox, oy, or_ = outer
    ix, iy, ir  = inner
    outer_poly = circle_to_poly(ox, oy, or_)
    inner_poly = circle_to_poly(ix, iy, ir)

    def norm(pts):
        return " ".join(f"{round(x/img_w,6)} {round(y/img_h,6)}" for x, y in pts)

    stem = f"{tid}_top_removed_bg_{tid}-top_removed_bg"
    (OUT_DIR / "labels" / f"{stem}.txt").write_text(f"0 {norm(outer_poly)}\n1 {norm(inner_poly)}\n")
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


def nothing(_): pass


def setup_trackbars(img_w, img_h, ox, oy, or_, ix, iy, ir_):
    try: cv2.destroyWindow("__tb__")
    except: pass
    cv2.namedWindow("__tb__", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("__tb__", 500, 250)
    cv2.createTrackbar("outer_x", "__tb__", int(ox),  img_w, nothing)
    cv2.createTrackbar("outer_y", "__tb__", int(oy),  img_h, nothing)
    cv2.createTrackbar("outer_r", "__tb__", int(or_), 900,   nothing)
    cv2.createTrackbar("inner_x", "__tb__", int(ix),  img_w, nothing)
    cv2.createTrackbar("inner_y", "__tb__", int(iy),  img_h, nothing)
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


WIN = "Correct (data_without) — S=Save  R=Reset  LEFT=Back  Q=Quit"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 900, 1000)

idx = 0
while 0 <= idx < len(bad_ids):
    tid      = bad_ids[idx]
    img_path = DATA_DIR / f"{tid}_top_removed_bg_{tid}-top_removed_bg.png"
    if not img_path.exists():
        print(f"  SKIP: {img_path} not found")
        idx += 1
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    if tid in corrected:
        auto_outer = corrected[tid]["outer"]
        auto_inner = corrected[tid]["inner"]
    else:
        ao, ai = detect_auto(img)
        auto_outer = ao if ao is not None else [w//2, h//2, 600]
        auto_inner = ai if ai is not None else [w//2, h//2, 380]

    setup_trackbars(w, h,
                    auto_outer[0], auto_outer[1], auto_outer[2],
                    auto_inner[0], auto_inner[1], auto_inner[2])

    print(f"\n[{idx+1}/{len(bad_ids)}] Correcting {tid}  (S=save, R=reset, LEFT=back, Q=quit)")

    while True:
        ox, oy, or_, ix, iy, ir_ = get_trackbars()
        preview = draw_preview(img, ox, oy, or_, ix, iy, ir_)

        cv2.rectangle(preview, (0, h-55), (w, h), (30, 30, 30), -1)
        cv2.putText(preview, f"[{idx+1}/{len(bad_ids)}] {tid}  outer r={or_}  inner r={ir_}",
                    (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(preview, "S=Save   R=Reset   LEFT=Back   Q=Quit",
                    (10, h-8),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow(WIN, preview)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            save_annotation(tid, w, h, (ox, oy, or_), (ix, iy, ir_))
            ann_img = draw_preview(img, ox, oy, or_, ix, iy, ir_)
            stem = f"{tid}_top_removed_bg_{tid}-top_removed_bg"
            cv2.imwrite(str(OUT_DIR / "annotated_images" / f"{stem}_annotated.png"), ann_img)
            print(f"  Saved: outer=({ox},{oy},r{or_})  inner=({ix},{iy},r{ir_})")
            idx += 1
            break
        elif key == ord('r'):
            setup_trackbars(w, h,
                            auto_outer[0], auto_outer[1], auto_outer[2],
                            auto_inner[0], auto_inner[1], auto_inner[2])
        elif key == 81:
            idx = max(0, idx - 1)
            break
        elif key in (ord('q'), 27):
            idx = len(bad_ids)
            break

cv2.destroyAllWindows()
print(f"\nCorrected {len(corrected)} images. Run 3_merge_without.py when done.")