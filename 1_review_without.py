"""
STEP 1 (data_without) — Quick review of annotated background-removed images.

Controls:
  K or RIGHT  -> keep (good)
  B           -> flag as bad
  LEFT        -> go back one image
  Q or ESC    -> quit and save progress
"""

import cv2
import os
import glob

ANN_DIR   = r"c:\Users\lojai\Documents\labelling\data_without_output\final\annotated_images"
BAD_FILE  = r"c:\Users\lojai\Documents\labelling\bad_images_without.txt"
DONE_FILE = r"c:\Users\lojai\Documents\labelling\review_progress_without.txt"

files = sorted(glob.glob(os.path.join(ANN_DIR, "*.png")))

done_ids = set()
bad_ids  = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                tid, verdict = parts
                done_ids.add(tid)
                if verdict == "bad":
                    bad_ids.add(tid)

def save_progress():
    with open(DONE_FILE, "w") as f:
        for tid in done_ids:
            verdict = "bad" if tid in bad_ids else "ok"
            f.write(f"{tid},{verdict}\n")
    with open(BAD_FILE, "w") as f:
        for tid in sorted(bad_ids):
            f.write(tid + "\n")

def get_tyre_id(path):
    return os.path.basename(path).split("_")[0]

start = 0
for i, f in enumerate(files):
    if get_tyre_id(f) not in done_ids:
        start = i
        break
else:
    start = len(files)

cv2.namedWindow("Review (data_without)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Review (data_without)", 900, 900)

idx   = start
total = len(files)

while 0 <= idx < total:
    f   = files[idx]
    tid = get_tyre_id(f)
    img = cv2.imread(f)
    h, w = img.shape[:2]

    already_done = tid in done_ids
    verdict_str  = ("BAD" if tid in bad_ids else "KEEP") if already_done else "?"
    color = (0, 0, 200) if (tid in bad_ids) else (0, 200, 0) if already_done else (200, 200, 200)

    banner = img.copy()
    cv2.rectangle(banner, (0, h - 60), (w, h), (30, 30, 30), -1)
    cv2.putText(banner, f"[{idx+1}/{total}]  ID: {tid}  |  {verdict_str}",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(banner, "K/RIGHT=Keep   B=Bad   LEFT=Back   Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("Review (data_without)", banner)
    key = cv2.waitKey(0) & 0xFF

    if key in (ord('k'), 83, ord('\r')):
        done_ids.add(tid)
        bad_ids.discard(tid)
        save_progress()
        idx += 1
    elif key == ord('b'):
        done_ids.add(tid)
        bad_ids.add(tid)
        save_progress()
        idx += 1
    elif key == 81:
        idx = max(0, idx - 1)
    elif key in (ord('q'), 27):
        break

cv2.destroyAllWindows()
save_progress()

remaining = total - len(done_ids)
print(f"\nReview complete: {len(done_ids)}/{total} reviewed")
print(f"Flagged as bad : {len(bad_ids)}")
print(f"Remaining      : {remaining}")
if bad_ids:
    print(f"Bad IDs saved to: {BAD_FILE}")