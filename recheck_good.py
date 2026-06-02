"""
Recheck good images — shows only images marked as KEEP so far.

Change GROUP to whichever group you want to recheck (1, 2, 3, or 4).
Press B to flip a good image to bad, K/RIGHT to confirm it stays good.
Progress is saved back to the same group file so it all stays in sync.

Controls:
  K or RIGHT  -> confirm keep
  B           -> flip to bad
  LEFT        -> go back
  Q or ESC    -> quit and save
"""

import cv2, os, glob
from pathlib import Path

GROUP = 4   # <-- change to 1, 2, 3, or 4

ANN_DIR   = r"c:\Users\lojai\Documents\labelling\the_data_output\final\annotated_images"
DONE_FILE = Path(rf"c:\Users\lojai\Documents\labelling\review_progress_data_g{GROUP}.txt")
BAD_FILE  = Path(rf"c:\Users\lojai\Documents\labelling\bad_images_data_g{GROUP}.txt")

# Load existing progress
done_ids = {}   # stem -> "ok" or "bad"
if DONE_FILE.exists():
    for line in DONE_FILE.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) == 2:
            done_ids[parts[0]] = parts[1]

bad_ids = {k for k, v in done_ids.items() if v == "bad"}
good_ids = {k for k, v in done_ids.items() if v == "ok"}

# Only show the good ones
all_files = sorted(glob.glob(os.path.join(ANN_DIR, "*.png")))
files = [f for f in all_files
         if os.path.basename(f).replace("_annotated.png", "") in good_ids]

print(f"Group {GROUP} — {len(good_ids)} good images to recheck")

def get_stem(path):
    return os.path.basename(path).replace("_annotated.png", "")

def save_progress():
    with open(DONE_FILE, "w") as f:
        for stem, verdict in done_ids.items():
            f.write(f"{stem},{verdict}\n")
    with open(BAD_FILE, "w") as f:
        for stem in sorted(bad_ids):
            f.write(stem + "\n")
    # Update combined bad file
    combined = set()
    for g in range(1, 5):
        p = Path(rf"c:\Users\lojai\Documents\labelling\bad_images_data_g{g}.txt")
        if p.exists():
            combined |= {l.strip() for l in p.read_text().splitlines() if l.strip()}
    if combined:
        Path(r"c:\Users\lojai\Documents\labelling\bad_images_data.txt").write_text(
            "\n".join(sorted(combined)) + "\n"
        )

if not files:
    print("No good images found for this group yet.")
    exit()

cv2.namedWindow("Recheck Good", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Recheck Good", 900, 900)

idx   = 0
total = len(files)

while 0 <= idx < total:
    f    = files[idx]
    stem = get_stem(f)
    img  = cv2.imread(f)
    h, w = img.shape[:2]

    banner = img.copy()
    cv2.rectangle(banner, (0, h-60), (w, h), (30, 30, 30), -1)
    cv2.putText(banner, f"[{idx+1}/{total}]  {stem}",
                (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(banner, "K/RIGHT=Confirm Keep   B=Flip to Bad   LEFT=Back   Q=Quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Recheck Good", banner)
    key = cv2.waitKey(0) & 0xFF

    if key in (ord('k'), 83, ord('\r')):
        idx += 1
    elif key == ord('b'):
        done_ids[stem] = "bad"
        bad_ids.add(stem)
        good_ids.discard(stem)
        save_progress()
        files.pop(idx)
        total -= 1
    elif key == 81:
        idx = max(0, idx - 1)
    elif key in (ord('q'), 27):
        break

cv2.destroyAllWindows()
save_progress()

flipped = len([k for k, v in done_ids.items() if v == "bad"]) - len(bad_ids) + len(
    [stem for stem in good_ids if done_ids.get(stem) == "bad"]
)
print(f"\nDone. Good remaining: {len(good_ids)}  |  Total bad in group {GROUP}: {len(bad_ids)}")