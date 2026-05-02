"""Shared helpers: model load, single-image inference, output dir."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_IMAGE = (
    "/home/rohan-garg/Downloads/ASL RGB NAV GRAPH/longrange/"
    "highrange_rgb_nav_graph_dataset_2024-11-11-16-14-23_mission/rgb_000000.png"
)
DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


@dataclass
class DA3Result:
    image: np.ndarray      # [H, W, 3] uint8, the model's processed image
    depth: np.ndarray      # [H, W] float32 — metric (m) for NESTED/METRIC, else relative
    conf:  np.ndarray      # [H, W] float32
    K:     np.ndarray      # [3, 3] float32, intrinsics matched to `image`
    is_metric: bool


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", default=DEFAULT_IMAGE, help="path to an RGB image")
    p.add_argument("--model", default=DEFAULT_MODEL, help="DA3 HF model id")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=None, help="defaults to ./outputs/<image-stem>")


def resolve_out_dir(image: str, out_dir: str | None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent / "outputs" / Path(image).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def is_metric_checkpoint(model_id: str) -> bool:
    name = model_id.upper()
    return "NESTED" in name or "METRIC" in name


def run_da3(image_path: str, model_id: str, device: str) -> DA3Result:
    """Load DA3, run inference on one image, return the first prediction."""
    # Imported lazily so the script can `--help` without DA3 installed.
    from depth_anything_3.api import DepthAnything3

    print(f"[da3] loading {model_id} on {device} ...")
    model = DepthAnything3.from_pretrained(model_id).to(device=torch.device(device))
    model.eval()

    print(f"[da3] running inference on {image_path}")
    with torch.no_grad():
        pred = model.inference([image_path])

    image = np.asarray(pred.processed_images[0])  # [H, W, 3] uint8
    depth = np.asarray(pred.depth[0]).astype(np.float32)
    conf  = np.asarray(pred.conf[0]).astype(np.float32)
    K     = np.asarray(pred.intrinsics[0]).astype(np.float32)

    return DA3Result(
        image=image, depth=depth, conf=conf, K=K,
        is_metric=is_metric_checkpoint(model_id),
    )
