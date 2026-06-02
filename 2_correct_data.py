"""
STEP 2 (the_data) — Manually correct circles for flagged images.

Change BATCH to work through 100 images at a time.
BATCH = 1 -> images 1-100
BATCH = 2 -> images 101-200  etc.

Controls:
  S        -> save and go to next
  R        -> reset to auto-detected values
  LEFT     -> go back
  Q / ESC  -> quit
"""

import cv2, numpy as np, json, math
from pathlib import Path

BATCH      = 2    # <-- change to 1, 2, 3 ... to work in batches of 100
BATCH_SIZE = 100

ROOT      = Path(r"c:\Users\lojai\Documents\labelling\the_data")
BAD_FILE  = Path(r"c:\Users\lojai\Documents\labelling\bad_images_data.txt")
OUT_DIR   = Path(r"c:\Users\lojai\Documents\labelling\the_data_output\corrected")
DONE_FILE = Path(r"c:\Users\lojai\Documents\labelling\correct_progress_data.json")
ANN_JSON  = Path(r"c:\Users\lojai\Documents\labelling\the_data_output\inner_and_outer\annotations.json")

(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)

POLY_POINTS = 72

if not BAD_FILE.exists() or not BAD_FILE.read_text().strip():
    print("No bad images to correct.")
    exit()

all_bad_ids   = [l.strip() for l in BAD_FILE.read_text().splitlines() if l.strip()]
corrected_so_far = json.loads(DONE_FILE.read_text()) if DONE_FILE.exists() else {}
remaining_ids = [s for s in all_bad_ids if s not in corrected_so_far]
start   = (BATCH - 1) * BATCH_SIZE
end     = start + BATCH_SIZE
bad_ids = remaining_ids[start:end]
print(f"Batch {BATCH}: {len(bad_ids)} images to correct  |  {len(corrected_so_far)} already done  |  {len(remaining_ids)} remaining total")

corrected = {}
if DONE_FILE.exists():
    corrected = json.loads(DONE_FILE.read_text())

# Build lookup: stem -> (folder, tyre_id)
with open(ANN_JSON) as f:
    coco = json.load(f)
stem_lookup = {
    img["file_name"].replace(".png", ""): (img.get("folder", "4"), str(img["tyre_id"]))
    for img in coco["images"]
}
# bad_ids already set above (batch slice) — do not overwrite


def detect_auto(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    d = min(w, h)
    blurred = cv2.GaussianBlur(gray, (11,11), 3)
    cx, cy = w/2, h/2
    def best(mn, mx, p2):
        c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                             param1=80, param2=p2, minRadius=int(mn), maxRadius=int(mx))
        if c is None: return None
        return sorted(c[0], key=lambda x: abs(x[0]-cx)+abs(x[1]-cy))[0]
    outer = best(d*0.30, d*0.60, 50)
    inner = best(d*0.15, d*0.38, 60)
    return (list(outer) if outer is not None else None,
            list(inner) if inner is not None else None)


def circle_to_poly(cx, cy, r, n=POLY_POINTS):
    angles = np.linspace(0, 2*math.pi, n, endpoint=False)
    return [(float(cx+r*math.cos(a)), float(cy+r*math.sin(a))) for a in angles]


def save_annotation(stem, img_w, img_h, outer, inner):
    ox, oy, or_ = outer
    ix, iy, ir  = inner
    outer_poly = circle_to_poly(ox, oy, or_)
    inner_poly = circle_to_poly(ix, iy, ir)
    def norm(pts):
        return " ".join(f"{round(x/img_w,6)} {round(y/img_h,6)}" for x, y in pts)
    (OUT_DIR / "labels" / f"{stem}.txt").write_text(f"0 {norm(outer_poly)}\n1 {norm(inner_poly)}\n")
    corrected[stem] = {"outer": list(outer), "inner": list(inner)}
    DONE_FILE.write_text(json.dumps(corrected, indent=2))


def draw_preview(img, ox, oy, or_, ix, iy, ir_):
    overlay = img.copy()
    outer_pts = np.array(circle_to_poly(ox, oy, or_), dtype=np.int32)
    inner_pts = np.array(circle_to_poly(ix, iy, ir_), dtype=np.int32)
    cv2.fillPoly(overlay, [outer_pts], (255,100,0))
    cv2.fillPoly(overlay, [inner_pts], (0,120,255))
    canvas = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(canvas, [outer_pts], True, (255,180,0), 3)
    cv2.polylines(canvas, [inner_pts], True, (0,220,0), 3)
    return canvas


def nothing(_): pass

def setup_trackbars(img_w, img_h, ox, oy, or_, ix, iy, ir_):
    try: cv2.destroyWindow("__tb__")
    except: pass
    cv2.namedWindow("__tb__", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("__tb__", 500, 250)
    cv2.createTrackbar("outer_x", "__tb__", int(ox),  img_w,        nothing)
    cv2.createTrackbar("outer_y", "__tb__", int(oy),  img_h,        nothing)
    cv2.createTrackbar("outer_r", "__tb__", int(or_), img_w//2,     nothing)
    cv2.createTrackbar("inner_x", "__tb__", int(ix),  img_w,        nothing)
    cv2.createTrackbar("inner_y", "__tb__", int(iy),  img_h,        nothing)
    cv2.createTrackbar("inner_r", "__tb__", int(ir_), img_w//3,     nothing)

def get_trackbars():
    return tuple(cv2.getTrackbarPos(n, "__tb__")
                 for n in ["outer_x","outer_y","outer_r","inner_x","inner_y","inner_r"])


WIN = "Correct (the_data) — S=Save  R=Reset  LEFT=Back  Q=Quit"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 900, 1000)

idx = 0
while 0 <= idx < len(bad_ids):
    stem   = bad_ids[idx]
    folder, tyre_id = stem_lookup.get(stem, ("4", stem.split("_")[1]))
    img_path = ROOT / folder / f"{tyre_id}_top_{tyre_id}-top.png"
    tid = stem   # use stem as the correction key
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
        auto_outer = ao if ao is not None else [w//2, h//2, int(min(w,h)*0.45)]
        auto_inner = ai if ai is not None else [w//2, h//2, int(min(w,h)*0.25)]

    setup_trackbars(w, h,
                    auto_outer[0], auto_outer[1], auto_outer[2],
                    auto_inner[0], auto_inner[1], auto_inner[2])

    print(f"\n[{idx+1}/{len(bad_ids)}] Correcting {stem}  (S=save, R=reset, LEFT=back, Q=quit)")

    while True:
        ox, oy, or_, ix, iy, ir_ = get_trackbars()
        preview = draw_preview(img, ox, oy, or_, ix, iy, ir_)
        cv2.rectangle(preview, (0, h-55), (w, h), (30,30,30), -1)
        cv2.putText(preview, f"[{idx+1}/{len(bad_ids)}] {tid}  outer r={or_}  inner r={ir_}",
                    (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(preview, "S=Save   R=Reset   LEFT=Back   Q=Quit",
                    (10, h-8),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
        cv2.imshow(WIN, preview)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            save_annotation(stem, w, h, (ox,oy,or_), (ix,iy,ir_))
            ann_img = draw_preview(img, ox, oy, or_, ix, iy, ir_)
            cv2.imwrite(str(OUT_DIR/"annotated_images"/f"{stem}_annotated.png"), ann_img)
            print(f"  Saved: outer=({ox},{oy},r{or_})  inner=({ix},{iy},r{ir_})")
            idx += 1; break
        elif key == ord('r'):
            setup_trackbars(w, h,
                            auto_outer[0], auto_outer[1], auto_outer[2],
                            auto_inner[0], auto_inner[1], auto_inner[2])
        elif key == 81:
            idx = max(0, idx-1); break
        elif key in (ord('q'), 27):
            idx = len(bad_ids); break

cv2.destroyAllWindows()
print(f"\nCorrected {len(corrected)} images. Run 3_merge_data.py when done.")