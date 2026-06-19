"""
Flatten yolo_dataset_without (train/val/test split) into
yolo_dataset_no_background (flat: images/ + labels/ + dataset.yaml).
Original folder is NOT modified.
"""

import shutil
from pathlib import Path

SRC  = Path(r"E:\labeling\yolo_dataset_without")
DST  = Path(r"E:\labeling\yolo_dataset_no_background")

(DST / "images").mkdir(parents=True, exist_ok=True)
(DST / "labels").mkdir(parents=True, exist_ok=True)

copied = 0
for split in ["train", "val", "test"]:
    for img in (SRC / "images" / split).glob("*.png"):
        shutil.copy2(img, DST / "images" / img.name)
        lbl = SRC / "labels" / split / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, DST / "labels" / lbl.name)
        copied += 1

yaml = f"""# No-background tyre dataset — {copied} images, flat (no split)

path: {DST.as_posix()}
train: images
val:   images

nc: 2
names:
  0: tyre_outer
  1: tyre_inner_hole
"""
(DST / "dataset.yaml").write_text(yaml)

print(f"Done. Copied {copied} images to {DST}")
