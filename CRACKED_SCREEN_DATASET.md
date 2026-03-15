# Cracked Screen Dataset — YOLOv9 Compatibility Assessment

**Dataset:** [Cracked Mobile Screen Dataset (DataCluster Labs)](https://www.kaggle.com/datasets/dataclusterlabs/cracked-screen-dataset)  
**Use case:** Train YOLOv9 to detect **broken / cracked screen** (and related classes).

---

## 1. Dataset summary

| Aspect | Details |
|--------|--------|
| **Content** | Mobile phone images: **good**, **cracked-screen**, **damaged** (3 classes) |
| **Size** | 3000+ images (Kaggle/DataCluster); Roboflow public sample ~100 images |
| **Quality** | HD+ (e.g. 1920×1080), crowdsourced, varied lighting/viewpoints |
| **Source** | DataCluster Labs (India), 2020–2021 |

The same/similar data appears as:
- **Kaggle:** `dataclusterlabs/cracked-screen-dataset`
- **Roboflow:** [Mobile Phone Dataset](https://universe.roboflow.com/datacluster-labs-agryi/mobile-phone-dataset) — export in COCO, YOLO, PASCAL-VOC, TF-Record
- **GitHub:** [datacluster-labs/Cracked-Screen-Image-Dataset](https://github.com/datacluster-labs/Cracked-Screen-Image-Dataset) (sample)

---

## 2. Compatibility with this YOLOv9 repo

**Verdict: compatible**, but the **annotation format** must match what the loader expects.

### What this project accepts

1. **COCO JSON**  
   - `annotations/instances_train.json`, `instances_val.json`  
   - Images in `images/train/`, `images/val/`  
   → **Fully supported.** If the dataset comes as COCO (e.g. from Roboflow “COCO” export), use it as-is after placing files in the layout above.

2. **.txt labels (this repo’s format)**  
   - One `.txt` per image in `labels/train/`, `labels/val/`  
   - Each line = **class_id** + **normalized (x, y) point pairs** in `[0, 1]`  
   - Bounding box is computed as min/max of those points  
   - So each line should be: **class_id x_min y_min x_max y_max** (5 numbers = 2 corners), all normalized.

### Common mismatch: “standard” YOLO format

Many tools (Roboflow “YOLO”, LabelImg, etc.) export **one line per object** as:

```text
class_id x_center y_center width height
```

(all normalized 0–1). This repo’s loader **does not** interpret that as center + size; it treats the four numbers as **two (x, y) points**, so boxes would be wrong. You must convert.

---

## 3. What to do after download

### Step 1: Inspect the downloaded structure

After downloading from Kaggle (or Roboflow), check:

- Where the **images** are (e.g. `train/`, `valid/`, `test/`).
- Where the **labels** are and in **which format**:
  - **COCO:** one or more `.json` files (e.g. `_annotations.coco.json` or `instances_train.json`).
  - **YOLO:** one `.txt` per image; open one and see if each line has **5 numbers** (likely `class x_center y_center width height`) or **5 numbers** as `class x_min y_min x_max y_max`, or more (polygon).

### Step 2: Normalize to the expected layout

Target layout (same as in `CUSTOM_DATASET.md`):

```text
data/cracked_screen/   # or your path
├── images/
│   ├── train/
│   │   └── *.jpg
│   └── val/
│       └── *.jpg
└── either:
    ├── annotations/
    │   ├── instances_train.json
    │   └── instances_val.json
    └── or labels/
        ├── train/
        │   └── *.txt
        └── val/
            └── *.txt
```

- If you have **COCO**: put the JSON(s) in `annotations/` and name them `instances_train.json` / `instances_val.json` (or point `train`/`validation` in the config to the split names that match your JSON filenames). Put images in `images/train/` and `images/val/`.
- If you have **standard YOLO .txt** (class, x_center, y_center, width, height): use the conversion script below to produce the **xyxy-style .txt** format and place them under `labels/train/` and `labels/val/`.

### Step 3: Convert standard YOLO .txt (if needed)

If each line in your `.txt` is `class_id x_center y_center width height` (normalized), run:

```bash
# If your download has train/ and val/ (or valid/) with labels inside:
python tools/convert_yolo_xywh_to_xyxy_txt.py \
  --labels-dir path/to/downloaded/dataset \
  --output-dir data/cracked_screen/labels \
  --splits train val

# If labels are in train/labels/ and valid/labels/, the script looks there automatically.
```

This writes one line per object as `class_id x_min y_min x_max y_max` (normalized), which this repo’s loader expects. Copy or link images into `data/cracked_screen/images/train/` and `data/cracked_screen/images/val/` to match.

### Step 4: Dataset config

Use the provided config that defines the 3 classes:

```bash
python yolo/lazy.py task=train dataset=cracked_screen
```

Config file: `yolo/config/dataset/cracked_screen.yaml` (path, train/val names, `class_num: 3`, `class_list: ['good', 'cracked-screen', 'damaged']`). Adjust `path` if your data is not under `data/cracked_screen`.

---

## 4. Recommended workflow

1. **Download** the dataset from Kaggle (or Roboflow “COCO” / “YOLO”).
2. **Identify** annotation format (COCO vs YOLO, and if YOLO then xywh vs xyxy).
3. **Arrange** images and labels into the layout above; if labels are standard YOLO xywh, run `convert_yolo_xywh_to_xyxy_txt.py`.
4. **Point** `path` in `yolo/config/dataset/cracked_screen.yaml` to your dataset root.
5. **Train:**  
   `python yolo/lazy.py task=train dataset=cracked_screen task.data.batch_size=8`

If you export from **Roboflow**, choosing **“COCO”** avoids the .txt conversion step; choosing **“YOLO”** usually gives xywh, so use the converter.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| Is the cracked-screen dataset suitable for YOLOv9? | **Yes** — object detection with 3 classes, good for “broken screen” detection. |
| Is it compatible with this repo? | **Yes**, once annotations are in **COCO** or this repo’s **.txt** format (class + xy corners, normalized). |
| What to watch for? | Standard YOLO **xywh** .txt must be converted to **xyxy-style** .txt (or use COCO). |
| Config to use? | `dataset=cracked_screen` with `yolo/config/dataset/cracked_screen.yaml`. |
