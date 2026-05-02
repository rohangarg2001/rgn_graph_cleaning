"""Run DA3 on one image and save a colourised depth map next to the RGB."""
from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from da3_common import add_common_args, resolve_out_dir, run_da3


def colorize(depth: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """Map depth → uint8 RGB via TURBO. Low-confidence pixels are blacked out."""
    valid = np.isfinite(depth)
    if conf is not None:
        valid &= conf > np.quantile(conf[np.isfinite(conf)], 0.05)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo, hi = np.quantile(depth[valid], [0.02, 0.98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (plt.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    out = resolve_out_dir(args.image, args.out_dir)

    res = run_da3(args.image, args.model, args.device)

    depth_rgb = colorize(res.depth, res.conf)
    side = np.concatenate([res.image, depth_rgb], axis=1)

    cv2.imwrite(str(out / "depth_color.png"), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "rgb_vs_depth.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    np.save(out / "depth_raw.npy", res.depth)
    np.save(out / "conf.npy", res.conf)

    finite = res.depth[np.isfinite(res.depth)]
    unit = "m" if res.is_metric else "(relative)"
    print(f"[depth] min/median/max = {finite.min():.3f}/{np.median(finite):.3f}/{finite.max():.3f} {unit}")
    print(f"[depth] wrote {out}/depth_color.png and rgb_vs_depth.png")


if __name__ == "__main__":
    main()
