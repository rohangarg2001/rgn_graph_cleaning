#!/usr/bin/env python3
"""
calibration_check.py — Diagnose Depth Anything calibration on a single frame.

Runs DA on ONE rgb_*.png, samples the depth window for every active node
in the matching graph_*.json, fits the calibration, prints a per-node
residual table, and saves a depth visualization (regardless of whether
any nodes would be flagged).

Read-only: never modifies the graph JSON.

Usage
─────
  python calibration_check.py \\
      --mission /path/to/<mission_dir> \\
      --frame   002100

  python calibration_check.py \\
      --mission ... --frame 002100 --calibrate linear        # try alternative
  python calibration_check.py \\
      --mission ... --frame 002100 --vis-out /tmp/check.png  # custom output
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from occlusion_clean import (
    OUTDOOR_MODEL, INDOOR_MODEL,
    DepthEstimator,
    AGG_CHOICES, CALIBRATE_CHOICES,
    sample_depth, _calibrate, _format_calibration,
    save_depth_visualization,
)


def main():
    ap = argparse.ArgumentParser(
        description="DA calibration diagnostic for a single frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mission",   type=Path, required=True, metavar="DIR",
                    help="Mission directory containing rgb_*.png and graph_*.json.")
    ap.add_argument("--frame",     required=True, metavar="IDX",
                    help='Frame index, e.g. "002100".')
    ap.add_argument("--window",    type=int, default=7, metavar="N")
    ap.add_argument("--agg",       choices=AGG_CHOICES, default="adaptive")
    ap.add_argument("--calibrate", choices=CALIBRATE_CHOICES, default="inverse")
    ap.add_argument("--abs-tol",   type=float, default=0.15, metavar="M",
                    help="For displayed flag column only (does not mutate).")
    ap.add_argument("--rel-tol",   type=float, default=0.15, metavar="F")
    ap.add_argument("--indoor",    action="store_true")
    ap.add_argument("--device",    type=int, default=None, metavar="ID")
    ap.add_argument("--vis-out",   type=Path, default=None, metavar="PATH",
                    help="Output PNG. Default: <mission>/_calib_check_<frame>.png")
    args = ap.parse_args()

    mp = args.mission
    if not mp.is_dir():
        sys.exit(f"[error] not a directory: {mp}")

    rgb_path   = mp / f"rgb_{args.frame}.png"
    graph_path = mp / f"graph_{args.frame}.json"
    if not rgb_path.exists() or not graph_path.exists():
        sys.exit(f"[error] missing rgb or graph for frame {args.frame} under {mp}")

    g     = json.loads(graph_path.read_text())
    nodes = g.get("nodes", [])
    print(f"\n  mission : {mp.name}")
    print(f"  frame   : {args.frame}")
    print(f"  active nodes (excluding any prior cleaning): {len(nodes)}")

    model_name = INDOOR_MODEL if args.indoor else OUTDOOR_MODEL
    est = DepthEstimator(model_name, device_arg=args.device)

    print(f"\n  Running depth estimation …")
    depth = est(rgb_path)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    print(f"  depth map: shape={depth.shape}, "
          f"finite range={finite.min():.2f} – {finite.max():.2f} m")

    # Sample raw predictions
    half = args.window // 2
    H, W = depth.shape
    raw: list = []  # (node, actual, raw_pred)
    for n in nodes:
        u, v = int(n["pixel"][0]), int(n["pixel"][1])
        if not (0 <= u < W and 0 <= v < H):
            continue
        actual = float(n["position_cam"][0])
        if actual <= 0:
            continue
        pred = sample_depth(depth, u, v, half, args.agg)
        if pred is None:
            continue
        raw.append((n, actual, pred))

    print(f"  in-frame, in-front-of-camera nodes tested: {len(raw)}")

    if not raw:
        sys.exit("[error] no testable nodes")

    # Calibrate
    actuals = np.array([a for _, a, _ in raw], dtype=np.float64)
    preds   = np.array([p for _, _, p in raw], dtype=np.float64)
    params, apply = _calibrate(preds, actuals, args.calibrate)
    print(f"\n  {_format_calibration(params)}")

    # Per-node table
    print()
    print(f"  {'ID':>5} {'TYPE':<11} {'PIXEL':<13} "
          f"{'ACTUAL':>7} {'RAW':>6} {'CORR':>6} {'RES':>7} {'WOULDFLAG':>10}")
    print(f"  {'-'*78}")
    residuals = []
    n_flagged = 0
    for n, actual, raw_p in sorted(raw, key=lambda r: r[1]):
        corr = float(apply(raw_p))
        res  = corr - actual
        residuals.append(res)
        tol  = max(args.abs_tol, args.rel_tol * corr)
        flag = "yes" if actual > corr + tol else ""
        if flag:
            n_flagged += 1
        pix = f"({n['pixel'][0]},{n['pixel'][1]})"
        print(f"  {n['id']:>5} {n.get('type','?'):<11} {pix:<13} "
              f"{actual:>7.2f} {raw_p:>6.2f} {corr:>6.2f} {res:>+7.2f} {flag:>10}")

    res_arr = np.array(residuals)
    print(f"  {'-'*78}")
    print(f"  median |residual| : {np.median(np.abs(res_arr)):.3f} m")
    print(f"  mean   |residual| : {np.mean(np.abs(res_arr)):.3f} m")
    print(f"  max    |residual| : {np.max(np.abs(res_arr)):.3f} m   "
          f"(at node id {raw[int(np.argmax(np.abs(res_arr)))][0]['id']})")
    print(f"  would-flag count  : {n_flagged} / {len(raw)}   "
          f"(tol = max({args.abs_tol} m, {args.rel_tol} * corr))")

    # Save depth vis
    out = (args.vis_out
           if args.vis_out is not None
           else mp / f"_calib_check_{args.frame}.png")
    node_preds = [(n, float(apply(p))) for n, _, p in raw]
    occluded   = {n["id"] for n, actual, p in raw
                  if actual > apply(p) + max(args.abs_tol, args.rel_tol * apply(p))}
    save_depth_visualization(depth, out, node_preds, occluded, cal_params=params)
    print(f"\n  saved depth vis : {out}")


if __name__ == "__main__":
    main()
