# Depth Anything V3 — test scripts

Inference scripts for running [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
on a single RGB frame from the long-range nav-graph dataset (e.g.
`/home/rohan-garg/Downloads/ASL RGB NAV GRAPH/longrange/highrange_rgb_nav_graph_dataset_2024-11-11-16-14-23_mission/rgb_000000.png`).

Three scripts:

- [run_depth.py](run_depth.py) — colorised depth map (PNG side-by-side).
- [run_pointcloud.py](run_pointcloud.py) — back-projects depth → coloured 3D point cloud, opens an Open3D viewer and writes a `.ply`.
- [run_segmentation.py](run_segmentation.py) — 2D segmentation overlay derived from depth (near / mid / far / sky bands).

All three share `da3_common.py` (image loading, model load, prediction).

---

## Requirements

### System
- **Python ≥ 3.10** (the DA3 `app` extra needs 3.10+; the core API works on 3.9 too but pin 3.10 to be safe).
- **CUDA-capable GPU** strongly recommended. Largest checkpoint (`DA3NESTED-GIANT-LARGE-1.1`, 1.4 B params) needs ~12 GB VRAM at fp16; the `LARGE` (0.35 B) runs comfortably on 6 GB; `BASE`/`SMALL` run on CPU but slowly.
- CUDA 11.8 / 12.x runtime (whatever your `torch` build was compiled against).

### Python packages

```bash
# 1. Pull the DA3 source (no PyPI release yet — it has to be installed from git):
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install "torch>=2" torchvision xformers
pip install -e .

# 2. Extra deps used by these test scripts:
pip install opencv-python numpy matplotlib open3d pillow huggingface_hub
```

(`open3d` is only needed by `run_pointcloud.py`.)

### Model weights

Weights live on Hugging Face under the `depth-anything/` namespace and are downloaded on first use by `from_pretrained(...)`. You do **not** need to download them manually — but you do need either:

- `huggingface-cli login` (if any model becomes gated), **or**
- internet access on first run so `huggingface_hub` can cache them under `~/.cache/huggingface/hub/`.

| Checkpoint                          | Params | Output         | Notes                                                  |
| ----------------------------------- | -----: | -------------- | ------------------------------------------------------ |
| `depth-anything/DA3-SMALL`          |  0.08B | relative depth | fastest; CPU-friendly                                  |
| `depth-anything/DA3-BASE`           |  0.12B | relative depth |                                                        |
| `depth-anything/DA3-LARGE-1.1`      |  0.35B | relative depth | good speed/quality default                             |
| `depth-anything/DA3-GIANT-1.1`      |  1.15B | relative depth |                                                        |
| `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 1.40B | **metric depth (m)** | recommended for outdoor scenes; needs ~12 GB VRAM |
| `depth-anything/DA3METRIC-LARGE`    |  0.35B | metric depth   | needs `metric = focal_px * net_out / 300` rescaling    |
| `depth-anything/DA3MONO-LARGE`      |  0.35B | relative depth | monocular-only (no multi-view)                         |

The scripts default to `DA3NESTED-GIANT-LARGE-1.1` because it produces metric depth out of the box, which makes the point cloud's units physically meaningful. Override with `--model` if you don't have the VRAM:

```bash
python run_depth.py --model depth-anything/DA3-LARGE-1.1
```

Note: only the `*METRIC*` and `*NESTED*` checkpoints give true metric depth. With the others, the depth and the point cloud are scale-ambiguous (visually fine, geometrically up to a global scale).

---

## Usage

```bash
cd depth_anything_v3_test_scripts
IMG="/home/rohan-garg/Downloads/ASL RGB NAV GRAPH/longrange/highrange_rgb_nav_graph_dataset_2024-11-11-16-14-23_mission/rgb_000000.png"

python run_depth.py        --image "$IMG"
python run_pointcloud.py   --image "$IMG"
python run_segmentation.py --image "$IMG"
```

Outputs land in `./outputs/<image-stem>/`:
- `depth_color.png`, `depth_raw.npy`
- `pointcloud.ply` (+ Open3D window if `$DISPLAY` is set)
- `segmentation.png`, `seg_mask.npy`
