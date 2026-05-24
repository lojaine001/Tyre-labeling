"""
STEP 5 — Combine both final datasets into one merged YOLO dataset.

Sources:
  data_with_output/final/      -> 154 images (with background)
  data_without_output/final/   -> 155 images (background removed)

Output:
  merged_dataset/
    images/train|val|test/
    labels/train|val|test/
    annotated_images/
    dataset.yaml

Train/val/test split: 70% / 15% / 15% (stratified so both sources appear in each split)
"""

import json
import shutil
import random
from pathlib import Path

WITH_FINAL    = Path(r"c:\Users\lojai\Documents\labelling\data_with_output\final")
WITHOUT_FINAL = Path(r"c:\Users\lojai\Documents\labelling\data_without_output\final")
DATA_WITH     = Path(r"c:\Users\lojai\Documents\labelling\data_with")
DATA_WITHOUT  = Path(r"c:\Users\lojai\Documents\labelling\data_without")
OUT_DIR       = Path(r"c:\Users\lojai\Documents\labelling\merged_dataset")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining

random.seed(42)

for split in ("train", "val", "test"):
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
(OUT_DIR / "annotated_images").mkdir(parents=True, exist_ok=True)

def split_list(items):
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train = items[:n_train]
    val   = items[n_train:n_train + n_val]
    test  = items[n_train + n_val:]
    return train, val, test


def load_entries(final_dir, src_images_dir, filename_pattern):
    """Returns list of (stem, original_image_path, label_path, annotated_path)."""
    entries = []
    labels_dir = final_dir / "labels"
    ann_dir    = final_dir / "annotated_images"
    for lbl in sorted(labels_dir.glob("*.txt")):
        stem    = lbl.stem                        # e.g. "90357_top_90357-top"
        pattern = filename_pattern(stem)          # original image filename
        img_src = src_images_dir / pattern
        ann_src = ann_dir / f"{stem}_annotated.png"
        if img_src.exists() and lbl.exists():
            entries.append((stem, img_src, lbl, ann_src))
    return entries


with_entries     = load_entries(
    WITH_FINAL,
    DATA_WITH,
    lambda stem: f"{stem}.png"
)
without_entries  = load_entries(
    WITHOUT_FINAL,
    DATA_WITHOUT,
    lambda stem: f"{stem}.png"
)

print(f"data_with entries    : {len(with_entries)}")
print(f"data_without entries : {len(without_entries)}")

# Shuffle each source independently so both appear in all splits
random.shuffle(with_entries)
random.shuffle(without_entries)

with_train,    with_val,    with_test    = split_list(with_entries)
without_train, without_val, without_test = split_list(without_entries)

splits = {
    "train": with_train + without_train,
    "val":   with_val   + without_val,
    "test":  with_test  + without_test,
}

total_copied = 0

for split_name, entries in splits.items():
    random.shuffle(entries)  # mix both sources within each split
    for stem, img_src, lbl_src, ann_src in entries:
        # Original image
        dst_img = OUT_DIR / "images" / split_name / img_src.name
        shutil.copy2(img_src, dst_img)

        # Label
        dst_lbl = OUT_DIR / "labels" / split_name / lbl_src.name
        shutil.copy2(lbl_src, dst_lbl)

        # Annotated image (all in one flat folder for easy browsing)
        if ann_src.exists():
            shutil.copy2(ann_src, OUT_DIR / "annotated_images" / ann_src.name)

        total_copied += 1

# Write dataset.yaml
total   = len(with_entries) + len(without_entries)
n_train = len(splits["train"])
n_val   = len(splits["val"])
n_test  = len(splits["test"])

yaml_content = f"""# Merged tyre dataset — with + without background
# Total: {total} images ({len(with_entries)} with background + {len(without_entries)} background removed)

path: {OUT_DIR.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: tyre_outer
  1: tyre_inner_hole
"""
(OUT_DIR / "dataset.yaml").write_text(yaml_content)

print(f"\nMerged dataset: {total} images total")
print(f"  Train : {n_train}")
print(f"  Val   : {n_val}")
print(f"  Test  : {n_test}")
print(f"\nOutput -> {OUT_DIR}")