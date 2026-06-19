"""
Split yolo_whole_data (flat) into train/val (80/20) with a fixed seed.
Output: yolo_whole_data_split/
          images/train/  images/val/
          labels/train/  labels/val/
          dataset.yaml
"""

import shutil, random
from pathlib import Path

SEED       = 42
VAL_RATIO  = 0.20
SRC        = Path(r"C:\Users\lojai\Documents\labelling\yolo_whole_data")
DST        = Path(r"C:\Users\lojai\Documents\labelling\yolo_whole_data_split")

for split in ["train", "val"]:
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

images = sorted((SRC / "images").glob("*.png"))
labels = {p.stem: p for p in (SRC / "labels").glob("*.txt")}

# Only keep images that have a matching label
paired = [img for img in images if img.stem in labels]
missing_labels = len(images) - len(paired)
if missing_labels:
    print(f"WARNING: {missing_labels} images have no label — excluded.")

random.seed(SEED)
random.shuffle(paired)

n_val   = int(len(paired) * VAL_RATIO)
val_set = set(img.stem for img in paired[:n_val])

n_train = 0
n_val_c = 0
for img in paired:
    split = "val" if img.stem in val_set else "train"
    shutil.copy2(img,              DST / "images" / split / img.name)
    shutil.copy2(labels[img.stem], DST / "labels" / split / (img.stem + ".txt"))
    if split == "train": n_train += 1
    else:                n_val_c += 1

yaml = f"""# Combined tyre dataset — split 80/20 (seed={SEED})

path: {DST.as_posix()}
train: images/train
val:   images/val

nc: 2
names:
  0: tyre_outer
  1: tyre_inner_hole
"""
(DST / "dataset.yaml").write_text(yaml)

print(f"Dataset split complete:")
print(f"  Total paired : {len(paired)}")
print(f"  Train        : {n_train}")
print(f"  Val          : {n_val_c}")
print(f"\nOutput -> {DST}")
