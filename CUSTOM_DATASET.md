# YOLOv9: Custom Photo Dataset — Prerequisites & Setup

This guide covers prerequisites and steps to train **YOLOv9** in this project with your own image dataset. It assumes you can use **WSL** for a Linux environment (recommended for CUDA).

---

## 1. Prerequisites

### 1.1 System & environment

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.8+ (3.10 recommended) |
| **OS** | Linux (native or **WSL2**) recommended; macOS (MPS/CPU); Windows + WSL2 for GPU |
| **GPU (optional)** | NVIDIA GPU + CUDA for faster training; CPU works for small experiments |

### 1.2 WSL (Windows)

- Use **WSL2** and ensure your project and data are under the WSL filesystem (e.g. `\\wsl$\Ubuntu\home\...`) for best performance.
- Install NVIDIA drivers on **Windows**; then install [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) so `nvidia-smi` works inside WSL.

### 1.3 Python dependencies

From the project root (in WSL or your environment):

```bash
cd /path/to/YOLO
pip install -r requirements.txt
```

Main packages: `torch`, `torchvision`, `hydra-core`, `opencv-python`, `Pillow`, `pycocotools`, `wandb`, etc.  
For **GPU**: install a CUDA-enabled PyTorch build (see [pytorch.org](https://pytorch.org/get-started/locally/)).

### 1.4 Pre-trained weights (transfer learning)

- Weights are resolved from the config (`weight: True` in `yolo/config/general.yaml`).
- You can set `weight=False` to train from scratch or `weight=/path/to/v9-c.pt` for a specific checkpoint.

---

## 2. Dataset layout

The code supports two label formats. Use **one** of the two structures below.

### Option A — COCO-style (JSON)

- **Annotations**: COCO-format JSON per split.
- **Naming**: `annotations/instances_<split>.json` where `<split>` matches the `train` / `validation` names in your dataset config (e.g. `instances_train.json`, `instances_val.json`).

```
DataSetRoot/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── train/
    │   ├── img1.jpg
    │   └── ...
    └── val/
        └── ...
```

### Option B — YOLO-style (.txt labels)

- **Images**: `images/<split>/` (e.g. `images/train/`, `images/val/`).
- **Labels**: One `.txt` per image in `labels/<split>/`, same base name as the image (e.g. `img1.jpg` → `labels/train/img1.txt`).
- **Txt format**: One line per object. Each line = **class_id** (integer) + **normalized coordinates** as pairs of (x, y) in range [0, 1]. The loader builds boxes from the min/max of these points, so you can use:
  - **Two corners**: `class_id x_min y_min x_max y_max` (5 numbers per line), or
  - **Polygon**: `class_id x1 y1 x2 y2 ...` (class_id + even number of floats).

Example for one object with two corners (normalized 0–1):

```
0 0.1 0.2 0.5 0.8
1 0.3 0.4 0.6 0.9
```

If your labels are in standard YOLO format (class_id, x_center, y_center, width, height), convert to (x_min, y_min, x_max, y_max) and write as above.

**Alternative**: You can provide a list file `DataSetRoot/<split>.txt` with one image path per line (relative to `DataSetRoot`). Then images and labels are resolved by replacing `images` with `labels` and `.jpg` with `.txt`.

---

## 3. Dataset config (YAML)

Create a dataset config (e.g. `yolo/config/dataset/custom.yaml`) that points to your root and class names:

```yaml
path: data/custom   # or absolute path, e.g. /home/user/datasets/mydata
train: train
validation: val

class_num: 3
class_list: ['ClassA', 'ClassB', 'ClassC']

# Omit auto_download for your own data
```

- **path**: Root directory of the dataset (as in the layout above).
- **train** / **validation**: Subfolder names under `images/` and `labels/` (or names used in `instances_<split>.json`).
- **class_num**: Number of classes.
- **class_list**: Class names (for logging/visualization); length must match `class_num`.
- Do **not** set `auto_download` if you are not using the built-in download feature.

---

## 4. Training commands

From the project root:

```bash
# Basic training with your custom dataset (use dataset=custom or your config name)
python yolo/lazy.py task=train dataset=custom use_wandb=True

# Smaller batch size if you run out of GPU memory
python yolo/lazy.py task=train dataset=custom task.data.batch_size=8

# YOLOv9-small for faster experiments
python yolo/lazy.py task=train dataset=custom model=v9-s task.data.batch_size=8

# From scratch (no pretrained weights)
python yolo/lazy.py task=train dataset=custom weight=False task.data.batch_size=8

# WSL / Linux with GPU
python yolo/lazy.py task=train dataset=custom device=cuda task.data.batch_size=8
```

Multi-GPU (e.g. 2 GPUs):

```bash
torchrun --nproc_per_node=2 yolo/lazy.py task=train dataset=custom device=[0,1]
```

Outputs (checkpoints, logs) go under `runs/train/<name>/` (see `yolo/config/config.yaml`: `hydra.run.dir`).

---

## 5. Quick checklist

- [ ] Python 3.8+ and `pip install -r requirements.txt`
- [ ] (Optional) WSL2 + NVIDIA driver + CUDA for GPU
- [ ] Dataset laid out as **Option A** (COCO JSON) or **Option B** (images + `.txt` labels)
- [ ] Dataset YAML created (e.g. `yolo/config/dataset/custom.yaml`) with correct `path`, `train`, `validation`, `class_num`, `class_list`
- [ ] Run: `python yolo/lazy.py task=train dataset=custom ...`

For more options (epochs, image size, optimizer, etc.) see the repo’s **HOWTO.md** and **docs/** (e.g. `docs/1_tutorials/0_allIn1.rst`).
