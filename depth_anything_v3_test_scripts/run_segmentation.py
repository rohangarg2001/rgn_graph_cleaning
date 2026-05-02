"""Depth-based 2D segmentation: split the scene into near / mid / far / sky bands."""
from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from da3_common import add_common_args, resolve_out_dir, run_da3

# Tab10 colours (RGB 0-255) for up to 10 segments.
_PALETTE = (np.array(plt.get_cmap("tab10").colors) * 255).astype(np.uint8)


def segment_by_depth(depth: np.ndarray, conf: np.ndarray,
                     n_bands: int = 3,
                     sky_quantile: float = 0.98,
                     sky_conf_quantile: float = 0.10) -> np.ndarray:
    """Return a uint8 label map: 0=invalid, 1..n_bands=near→far bands, n_bands+1=sky."""
    H, W = depth.shape
    labels = np.zeros((H, W), dtype=np.uint8)

    finite = np.isfinite(depth)
    if not finite.any():
        return labels

    conf_thr = np.quantile(conf[finite], sky_conf_quantile)
    depth_thr = np.quantile(depth[finite], sky_quantile)
    sky = finite & (conf < conf_thr) & (depth > depth_thr)

    # Quantile-based bands over the non-sky valid pixels.
    body = finite & ~sky
    if body.any():
        edges = np.quantile(depth[body], np.linspace(0, 1, n_bands + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        band_idx = np.digitize(depth, edges) - 1   # 0..n_bands-1
        labels[body] = band_idx[body].astype(np.uint8) + 1

    labels[sky] = n_bands + 1
    return labels


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for k in range(1, labels.max() + 1):
        rgb[labels == k] = _PALETTE[(k - 1) % len(_PALETTE)]
    return rgb


def overlay(image: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    mask = seg_rgb.any(axis=-1, keepdims=True)
    blended = (image.astype(np.float32) * (1 - alpha) +
               seg_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return np.where(mask, blended, image)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bands", type=int, default=3,
                    help="number of depth bands (excl. sky)")
    args = ap.parse_args()
    out = resolve_out_dir(args.image, args.out_dir)

    res = run_da3(args.image, args.model, args.device)
    labels = segment_by_depth(res.depth, res.conf, n_bands=args.bands)
    seg_rgb = colorize_labels(labels)
    overlaid = overlay(res.image, seg_rgb)

    side = np.concatenate([res.image, overlaid], axis=1)
    cv2.imwrite(str(out / "segmentation.png"), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "rgb_vs_segmentation.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    np.save(out / "seg_mask.npy", labels)

    counts = np.bincount(labels.ravel(), minlength=args.bands + 2)
    legend = ["invalid"] + [f"band{i+1}" for i in range(args.bands)] + ["sky"]
    print("[seg] pixel counts:")
    for name, c in zip(legend, counts):
        print(f"   {name:>8s}: {c:>10,d}")
    print(f"[seg] wrote {out}/segmentation.png")


if __name__ == "__main__":
    main()
