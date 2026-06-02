"""
STEP 1 (the_data) — Quick review of annotated images in groups.

Change GROUP below to 1, 2, 3, or 4 — each is ~450 images.
Progress is saved per group so you can stop and resume anytime.

Controls:
  K or RIGHT  -> keep (good)
  B           -> flag as bad
  LEFT        -> go back one image
  Q or ESC    -> quit and save progress
"""

import cv2, os, glob

GROUP = 4   # <-- change to 1, 2, 3, or 4

ANN_DIR   = r"c:\Users\lojai\Documents\labelling\the_data_output\final\annotated_images"
BAD_FILE  = rf"c:\Users\lojai\Documents\labelling\bad_images_data_g{GROUP}.txt"
DONE_FILE = rf"c:\Users\lojai\Documents\labelling\review_progress_data_g{GROUP}.txt"

all_files = sorted(glob.glob(os.path.join(ANN_DIR, "*.png")))
total_all = len(all_files)
size      = total_all // 4
starts    = {1: 0, 2: size, 3: size*2, 4: size*3}
ends      = {1: size, 2: size*2, 3: size*3, 4: total_all}
files     = all_files[starts[GROUP]:ends[GROUP]]
print(f"Group {GROUP}/4: images {starts[GROUP]+1}-{ends[GROUP]} of {total_all}")

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
            f.write(f"{tid},{'bad' if tid in bad_ids else 'ok'}\n")
    with open(BAD_FILE, "w") as f:
        for tid in sorted(bad_ids):
            f.write(tid + "\n")

def get_tyre_id(path):
    return os.path.basename(path).replace("_annotated.png", "")

start = 0
for i, f in enumerate(files):
    if get_tyre_id(f) not in done_ids:
        start = i
        break
else:
    start = len(files)

cv2.namedWindow("Review (the_data)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Review (the_data)", 900, 900)

idx   = start
total = len(files)

while 0 <= idx < total:
    f   = files[idx]
    tid = get_tyre_id(f)
    img = cv2.imread(f)
    h, w = img.shape[:2]

    already_done = tid in done_ids
    verdict_str  = ("BAD" if tid in bad_ids else "KEEP") if already_done else "?"
    color = (0,0,200) if tid in bad_ids else (0,200,0) if already_done else (200,200,200)

    banner = img.copy()
    cv2.rectangle(banner, (0, h-60), (w, h), (30,30,30), -1)
    cv2.putText(banner, f"[{idx+1}/{total}]  ID: {tid}  |  {verdict_str}",
                (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(banner, "K/RIGHT=Keep   B=Bad   LEFT=Back   Q=Quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    cv2.imshow("Review (the_data)", banner)
    key = cv2.waitKey(0) & 0xFF

    if key in (ord('k'), 83, ord('\r')):
        done_ids.add(tid); bad_ids.discard(tid); save_progress(); idx += 1
    elif key == ord('b'):
        done_ids.add(tid); bad_ids.add(tid); save_progress(); idx += 1
    elif key == 81:
        idx = max(0, idx - 1)
    elif key in (ord('q'), 27):
        break

cv2.destroyAllWindows()
save_progress()
print(f"\nGroup {GROUP} — Reviewed: {len(done_ids)}/{total}  |  Flagged bad: {len(bad_ids)}  |  Remaining: {total-len(done_ids)}")
if bad_ids:
    print(f"Bad IDs -> {BAD_FILE}")

# Merge all group bad files into one combined bad_images_data.txt
import pathlib
combined = set()
for g in range(1, 5):
    p = pathlib.Path(rf"c:\Users\lojai\Documents\labelling\bad_images_data_g{g}.txt")
    if p.exists():
        combined |= {l.strip() for l in p.read_text().splitlines() if l.strip()}
if combined:
    pathlib.Path(r"c:\Users\lojai\Documents\labelling\bad_images_data.txt").write_text(
        "\n".join(sorted(combined)) + "\n"
    )
    print(f"Combined bad list updated: {len(combined)} total bad images")