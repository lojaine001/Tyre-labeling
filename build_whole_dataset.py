"""
Combine all three datasets into one flat yolo_whole_data/ folder.
  - yolo_dataset_the_data      (1812 images, folder-prefixed names)
  - yolo_dataset_with_background (154 images)
  - yolo_dataset_no_background   (154 images)
Total: 2120 images, no file conflicts.
"""

import shutil
from pathlib import Path

SOURCES = [
    Path(r"C:\Users\lojai\Documents\labelling\yolo_dataset_the_data"),
    Path(r"E:\labeling\yolo_dataset_with_background"),
    Path(r"E:\labeling\yolo_dataset_no_background"),
]

DST = Path(r"C:\Users\lojai\Documents\labelling\yolo_whole_data")
(DST / "images").mkdir(parents=True, exist_ok=True)
(DST / "labels").mkdir(parents=True, exist_ok=True)

total = 0
for src in SOURCES:
    count = 0
    for img in (src / "images").glob("*.png"):
        shutil.copy2(img, DST / "images" / img.name)
        lbl = src / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, DST / "labels" / lbl.name)
        count += 1
    print(f"  {src.name}: {count} images copied")
    total += count

yaml = f"""# Combined tyre dataset — {total} images, flat (no split)

path: {DST.as_posix()}
train: images
val:   images

nc: 2
names:
  0: tyre_outer
  1: tyre_inner_hole
"""
(DST / "dataset.yaml").write_text(yaml)

print(f"\nDone. Total: {total} images -> {DST}")
