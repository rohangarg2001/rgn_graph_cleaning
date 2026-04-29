#!/usr/bin/env python3
"""
occlusion_clean.py — Depth-Anything-based occlusion filtering of nav-graph nodes.

For each rgb_*.png in each mission folder:
  1. Predict per-pixel metric depth with Depth Anything V2 Metric.
  2. For every active node in graph_*.json, aggregate predicted depth over a
     square window around the node's (u,v) pixel using --agg
     (min / p10 / p25 / median).  Default p25: the closest ~25% of pixels
     win, so thin/edge occluders (poles, car edges) reliably trigger the
     test, while a single noisy pixel does not.
  3. The node's true forward distance is position_cam[0] (x_fwd, since the
     dataset convention is camera_frame_x_fwd_y_down_z_left).
  4. If node_depth > predicted_depth + max(abs_tol, rel_tol * predicted_depth),
     the node is occluded → move it from "nodes" to "occlusion_cleaned_nodes",
     and any edge touching it from "edges" to "occlusion_cleaned_edges".

Re-runs are idempotent: each frame first restores its previous
occlusion_cleaned_* back into nodes/edges, then re-applies the test with
the current parameters. So tuning --window / --abs-tol / --rel-tol and
re-running will not let stale "bad" removals stick around.

Manual removals (removed_nodes / removed_edges from dataset_tool.py) are
never touched by this script.

Re-run with --revert to put everything back exactly as it was originally.

Usage
─────
  python occlusion_clean.py --mission-root DIR
  python occlusion_clean.py --mission-root DIR --missions 2024-10-01
  python occlusion_clean.py --mission-root DIR --window 9 --abs-tol 0.5 --rel-tol 0.15
  python occlusion_clean.py --mission-root DIR --indoor          # indoor metric model
  python occlusion_clean.py --mission-root DIR --vis-dir DIR     # save overlays
                                                                 #   AND a depth
                                                                 #   map per frame
                                                                 #   with removed
                                                                 #   nodes in magenta
  python occlusion_clean.py --mission-root DIR --agg min         # most aggressive
                                                                 #   removal
  python occlusion_clean.py --mission-root DIR --revert          # undo and exit
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUTDOOR_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
INDOOR_MODEL  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


# ── Mission discovery (matches dataset_tool.py) ───────────────────────────────

def get_missions(mission_root: Path, names: list | None) -> list[Path]:
    all_m = sorted(d for d in mission_root.iterdir()
                   if d.is_dir() and any(d.glob("overlay_[0-9]*.png")))
    if not names:
        return all_m
    out, seen = [], set()
    for pat in names:
        for m in all_m:
            if pat in m.name and m not in seen:
                out.append(m)
                seen.add(m)
    if not out:
        print(f"[warn] no missions matched: {names}")
    return out


def get_frame_indices(mp: Path) -> list[str]:
    return sorted(p.stem.split("_", 1)[1]
                  for p in mp.glob("rgb_[0-9]*.png"))


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2))


# ── Revert ────────────────────────────────────────────────────────────────────

def revert_mission(mp: Path) -> tuple[int, int, int, int]:
    n_files = n_restored = n_nodes = n_edges = 0
    for gp in sorted(mp.glob("graph_[0-9]*.json")):
        n_files += 1
        g = load_json(gp)
        on = g.pop("occlusion_cleaned_nodes", []) or []
        oe = g.pop("occlusion_cleaned_edges", []) or []
        if not on and not oe:
            continue
        g.setdefault("nodes", []).extend(on)
        g.setdefault("edges", []).extend(oe)
        save_json(gp, g)
        n_restored += 1
        n_nodes += len(on)
        n_edges += len(oe)
    return n_files, n_restored, n_nodes, n_edges


# ── Depth sampling ────────────────────────────────────────────────────────────

AGG_CHOICES = ("adaptive", "min", "p10", "p25", "median")

# How much closer the lower-tail (p25) must be vs the median for adaptive
# mode to declare a real foreground discontinuity in the window. Otherwise
# we treat the patch as smooth and use the unbiased median.
ADAPTIVE_GAP_ABS = 0.40   # metres
ADAPTIVE_GAP_REL = 0.12   # fraction of median depth


def sample_depth(depth: np.ndarray, u: int, v: int, half: int,
                 agg: str) -> float | None:
    """Aggregate depth over a (2*half+1) square window centred at (u,v).

    Modes:
      adaptive  — median when the window is smooth (e.g. flat ground, no
                  foreground object); p25 when there's a clear discontinuity
                  (a thin pole / car edge entered the window). This avoids
                  foreground-bleed false positives on close nodes while
                  still catching thin occluders.
      min/p10/p25 — always pick the lower tail. Aggressive; may flag close
                  nodes on smooth ground because the lower tail is biased
                  low when depth varies smoothly across the window.
      median    — always pick the centre. Conservative; misses thin
                  occluders that don't fill the window."""
    H, W = depth.shape
    u0, u1 = max(0, u - half), min(W, u + half + 1)
    v0, v1 = max(0, v - half), min(H, v + half + 1)
    if u1 <= u0 or v1 <= v0:
        return None
    patch = depth[v0:v1, u0:u1]
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if finite.size == 0:
        return None
    if agg == "min":
        return float(finite.min())
    if agg == "p10":
        return float(np.percentile(finite, 10))
    if agg == "p25":
        return float(np.percentile(finite, 25))
    if agg == "median":
        return float(np.median(finite))
    # adaptive
    med = float(np.median(finite))
    p25 = float(np.percentile(finite, 25))
    gap = med - p25
    if gap > max(ADAPTIVE_GAP_ABS, ADAPTIVE_GAP_REL * med):
        return p25
    return med


# ── Per-frame occlusion test ──────────────────────────────────────────────────

CALIBRATE_CHOICES = ("none", "bias", "linear", "inverse")


def _theil_sen(x: np.ndarray, y: np.ndarray,
               eps: float, slope_clip: tuple[float, float]
               ) -> tuple[float, float] | None:
    """Median-of-pairwise-slopes (Theil-Sen) regression of y on x.
    Returns (slope, intercept) or None if not enough usable pairs."""
    if x.size < 2:
        return None
    i, j = np.triu_indices(x.size, k=1)
    dx = x[j] - x[i]
    dy = y[j] - y[i]
    keep = np.abs(dx) > eps
    if not keep.any():
        return None
    s = float(np.median(dy[keep] / dx[keep]))
    s = float(np.clip(s, *slope_clip))
    b = float(np.median(y - s * x))
    return s, b


def _calibrate(preds: np.ndarray, actuals: np.ndarray,
               mode: str):
    """Returns (params_dict, apply_fn) where apply_fn(raw_pred) gives the
    corrected pred. Robust per-frame fit (Theil-Sen) using node depths.

      mode="none"    : identity.
      mode="bias"    : pred' = pred + median(actual - pred).
      mode="linear"  : pred' = s * pred + b   (linear in metric depth).
      mode="inverse" : 1/pred' = s' * (1/pred) + b'   (linear in inverse
                       depth — handles the common DA error pattern where
                       absolute error is large at close range and small at
                       far, which a single linear-in-depth fit cannot).

    Theil-Sen tolerates up to ~29%% outliers, so genuinely occluded nodes
    in the input set do not poison the calibration."""
    if mode == "none" or preds.size == 0:
        return {"mode": "none"}, (lambda x: x)

    if mode == "bias" or preds.size < 2:
        b = float(np.median(actuals - preds))
        return {"mode": "bias", "bias": b}, (lambda x, b=b: x + b)

    if mode == "linear":
        ts = _theil_sen(preds, actuals, eps=1e-3, slope_clip=(0.3, 3.0))
        if ts is not None:
            s, b = ts
            return ({"mode": "linear", "scale": s, "bias": b},
                    (lambda x, s=s, b=b: s * x + b))

    if mode == "inverse":
        valid = preds > 1e-3            # avoid 1/0
        if valid.sum() >= 2:
            ip = 1.0 / preds[valid]
            ia = 1.0 / actuals[valid]
            ts = _theil_sen(ip, ia, eps=1e-6, slope_clip=(0.1, 10.0))
            if ts is not None:
                s, b = ts
                def apply(x, s=s, b=b):
                    if x <= 1e-6:
                        return x
                    inv = s / x + b
                    return float(1.0 / inv) if inv > 1e-6 else float("inf")
                return ({"mode": "inverse", "inv_scale": s, "inv_bias": b},
                        apply)

    # Fallback: bias-only.
    b = float(np.median(actuals - preds))
    return {"mode": "bias", "bias": b}, (lambda x, b=b: x + b)


def _format_calibration(p: dict) -> str:
    m = p.get("mode", "none")
    if m == "none":
        return "DA calibration: none"
    if m == "bias":
        return f"DA calibration:  pred' = pred + {p['bias']:+.2f} m"
    if m == "linear":
        return (f"DA calibration:  pred' = {p['scale']:.3f} * pred "
                f"+ {p['bias']:+.2f} m")
    if m == "inverse":
        return (f"DA calibration:  1/pred' = {p['inv_scale']:.3f} / pred "
                f"+ {p['inv_bias']:+.4f}")
    return "DA calibration: ?"


def occlusion_clean_frame(graph_path: Path, depth: np.ndarray,
                          win: int, abs_tol: float, rel_tol: float,
                          agg: str, calibrate: str
                          ) -> tuple[list, list, list, set, dict]:
    """Returns (removed_nodes, removed_edges, node_preds, occluded_ids,
    cal_params).

    node_preds is a list of (node_dict, pred_depth_or_None) for every node
    that was tested (in-frame, in-front-of-camera). pred values already
    have per-frame calibration applied (formula varies with --calibrate),
    so the test reads `actual > pred + tol` directly."""
    g = load_json(graph_path)

    # Idempotent re-run: restore any prior occlusion_cleaned_* back into
    # nodes/edges so this run starts from a clean slate. Manual removed_*
    # (from dataset_tool.py) is never touched here.
    prev_n = g.pop("occlusion_cleaned_nodes", []) or []
    prev_e = g.pop("occlusion_cleaned_edges", []) or []
    if prev_n:
        g.setdefault("nodes", []).extend(prev_n)
    if prev_e:
        g.setdefault("edges", []).extend(prev_e)

    nodes = g.get("nodes", [])
    occluded_ids: set = set()
    node_preds: list = []
    cal_params: dict = {"mode": "none"}
    apply = lambda x: x  # identity by default

    if nodes:
        half = win // 2
        H, W = depth.shape

        # Pass 1: collect raw predictions for every in-frame node.
        raw_preds: list = []                # (node, raw_pred_or_None)
        for n in nodes:
            u, v = int(n["pixel"][0]), int(n["pixel"][1])
            if not (0 <= u < W and 0 <= v < H):
                continue
            node_depth = float(n["position_cam"][0])      # x_fwd = depth
            if node_depth <= 0:                            # behind the camera
                continue
            pred = sample_depth(depth, u, v, half, agg)
            raw_preds.append((n, pred))

        # Per-frame robust calibration.
        valid = [(float(n["position_cam"][0]), p)
                 for n, p in raw_preds if p is not None]
        if valid:
            actuals_arr = np.array([a for a, _ in valid], dtype=np.float64)
            preds_arr   = np.array([p for _, p in valid], dtype=np.float64)
            cal_params, apply = _calibrate(preds_arr, actuals_arr, calibrate)

        # Pass 2: apply the test using the calibrated prediction.
        for n, pred in raw_preds:
            if pred is None:
                node_preds.append((n, None))
                continue
            pred_c = apply(pred)
            node_preds.append((n, pred_c))
            node_depth = float(n["position_cam"][0])
            tol = max(abs_tol, rel_tol * pred_c)
            if node_depth > pred_c + tol:
                occluded_ids.add(n["id"])

    rem_n: list = []
    rem_e: list = []
    if occluded_ids:
        keep_n = [n for n in nodes if n["id"] not in occluded_ids]
        rem_n  = [n for n in nodes if n["id"] in occluded_ids]
        edges  = g.get("edges", [])
        keep_e = [e for e in edges
                  if e["node_id_0"] not in occluded_ids
                  and e["node_id_1"] not in occluded_ids]
        rem_e  = [e for e in edges
                  if e["node_id_0"] in occluded_ids
                  or e["node_id_1"] in occluded_ids]
        g["nodes"] = keep_n
        g["edges"] = keep_e
        g["occlusion_cleaned_nodes"] = rem_n
        g["occlusion_cleaned_edges"] = rem_e

    # Save if we changed anything — either by restoring previous state
    # or by flagging new occlusions (or both).
    if prev_n or prev_e or occluded_ids:
        save_json(graph_path, g)

    return rem_n, rem_e, node_preds, occluded_ids, cal_params


# ── Visualization ─────────────────────────────────────────────────────────────

VIS_COLOUR = (255, 0, 255)   # BGR magenta — pops on natural scenes


def save_visualization(overlay_src: Path, out_path: Path,
                       removed_nodes: list,
                       colour: tuple = VIS_COLOUR) -> bool:
    img = cv2.imread(str(overlay_src))
    if img is None:
        return False
    for n in removed_nodes:
        u, v = int(n["pixel"][0]), int(n["pixel"][1])
        # outer black ring + filled magenta + white X to read as "removed"
        cv2.circle(img, (u, v), 14, (0, 0, 0),       2, cv2.LINE_AA)
        cv2.circle(img, (u, v), 11, colour,         -1, cv2.LINE_AA)
        cv2.line  (img, (u - 6, v - 6), (u + 6, v + 6), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line  (img, (u - 6, v + 6), (u + 6, v - 6), (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def clear_mission_vis(mission_vis_dir: Path):
    """Wipe stale per-frame vis files from a previous run, so the saved set
    always reflects the current parameters."""
    if not mission_vis_dir.exists():
        return
    for p in mission_vis_dir.iterdir():
        if p.is_file():
            p.unlink()


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Colorize a metric depth map for human viewing.

    Clips to the [2nd, 98th] percentiles so a few extreme pixels don't
    flatten the rest of the image, then applies INFERNO (close=bright)."""
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(finite, (2, 98))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    norm = np.clip((depth - lo) / (hi - lo), 0, 1)
    # invert so closer = brighter = warmer colour
    norm_u8 = (255 * (1.0 - norm)).astype(np.uint8)
    coloured = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
    coloured[~np.isfinite(depth) | (depth <= 0)] = (0, 0, 0)
    return coloured


def save_depth_visualization(depth: np.ndarray, out_path: Path,
                             node_preds: list, occluded_ids: set,
                             cal_params: dict | None = None,
                             colour_occluded: tuple = VIS_COLOUR,
                             colour_active: tuple = (60, 220, 60)) -> bool:
    """Save a colorized depth map with each node drawn on top.

    Labels show `node_depth / calibrated_DA_depth` (metres). The calibration
    used for this frame is printed in the top-left corner so you can see
    whether DA is being adjusted heavily or barely at all."""
    img = _colorize_depth(depth)
    if img is None:
        return False
    fn = cv2.FONT_HERSHEY_SIMPLEX

    # Pass 1: small, semi-transparent filled markers (no outline ring).
    overlay = img.copy()
    for n, _ in node_preds:
        u, v = int(n["pixel"][0]), int(n["pixel"][1])
        col  = colour_occluded if n["id"] in occluded_ids else colour_active
        cv2.circle(overlay, (u, v), 4, col, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Pass 2: per-node labels — black text + thin white halo for legibility
    # on both bright (close) and dark (far) regions of the map.
    for n, pred in node_preds:
        u, v = int(n["pixel"][0]), int(n["pixel"][1])
        actual = float(n["position_cam"][0])
        label  = (f"{actual:.1f}/{pred:.1f}" if pred is not None
                  else f"{actual:.1f}/-")
        cv2.putText(img, label, (u + 6, v - 5), fn, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, label, (u + 6, v - 5), fn, 0.55, (0, 0, 0),       1, cv2.LINE_AA)

    # Calibration banner (top-left)
    cal_txt = _format_calibration(cal_params or {"mode": "none"})
    cv2.putText(img, cal_txt, (10, 30), fn, 0.65, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, cal_txt, (10, 30), fn, 0.65, (0, 0, 0),       1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


# ── Depth Anything wrapper ────────────────────────────────────────────────────

class DepthEstimator:
    """Direct AutoModelForDepthEstimation wrapper.

    We avoid `pipeline("depth-estimation")` because some pipeline post-
    processing normalises depth into [0,1]. For metric models we want the
    raw model output (already in metres)."""

    def __init__(self, model_name: str, device_arg: int | None):
        try:
            import torch  # type: ignore
            from transformers import (                # type: ignore
                AutoImageProcessor, AutoModelForDepthEstimation,
            )
        except ImportError as e:
            sys.exit(
                "missing dependency. Install with:\n"
                "  pip install torch transformers pillow\n"
                f"({e})"
            )

        self.torch = torch
        if device_arg is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_arg < 0:
            self.device = "cpu"
        else:
            self.device = f"cuda:{device_arg}"

        print(f"  Loading {model_name} on {self.device} …")
        t0 = time.time()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = (AutoModelForDepthEstimation
                      .from_pretrained(model_name)
                      .to(self.device)
                      .eval())
        print(f"  Loaded in {time.time() - t0:.1f}s")

    def __call__(self, rgb_path: Path) -> np.ndarray:
        from PIL import Image  # type: ignore

        img = Image.open(rgb_path).convert("RGB")
        W, H = img.size
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        # predicted_depth is (B, H', W') in metres for the metric variants
        depth = self.torch.nn.functional.interpolate(
            out.predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze().detach().cpu().numpy().astype(np.float32)
        return depth


# ── Per-mission processing ────────────────────────────────────────────────────

def process_mission(mp: Path, est: DepthEstimator,
                    win: int, abs_tol: float, rel_tol: float,
                    agg: str, calibrate: str,
                    vis_dir: Path | None = None,
                    save_overlay: bool = True,
                    save_depth: bool = True):
    indices = get_frame_indices(mp)
    print(f"\n  [{mp.name}]  {len(indices)} frames")

    mission_vis_dir = None
    if vis_dir is not None and (save_overlay or save_depth):
        mission_vis_dir = vis_dir / mp.name
        clear_mission_vis(mission_vis_dir)

    total_n = total_e = touched = vis_saved = 0
    t0 = time.time()
    for i, fidx in enumerate(indices, 1):
        rgb_path     = mp / f"rgb_{fidx}.png"
        graph_path   = mp / f"graph_{fidx}.json"
        overlay_path = mp / f"overlay_{fidx}.png"
        if not rgb_path.exists() or not graph_path.exists():
            continue
        depth = est(rgb_path)
        rem_n, rem_e, node_preds, occluded_ids, cal_params = occlusion_clean_frame(
            graph_path, depth, win, abs_tol, rel_tol, agg, calibrate)
        rn, re = len(rem_n), len(rem_e)
        total_n += rn
        total_e += re

        if mission_vis_dir is not None and rem_n:
            ok_overlay = ok_depth = False
            if save_overlay and overlay_path.exists():
                ok_overlay = save_visualization(
                    overlay_path,
                    mission_vis_dir / f"overlay_{fidx}_occluded.png",
                    rem_n,
                )
            if save_depth:
                ok_depth = save_depth_visualization(
                    depth,
                    mission_vis_dir / f"depth_{fidx}_occluded.png",
                    node_preds, occluded_ids,
                    cal_params=cal_params,
                )
            if ok_overlay or ok_depth:
                vis_saved += 1
        if rn or re:
            touched += 1
        if i % 20 == 0 or i == len(indices):
            dt = time.time() - t0
            rate = i / dt if dt else 0.0
            extra = f"  vis: {vis_saved}" if mission_vis_dir is not None else ""
            print(f"    {i:4d}/{len(indices)}  "
                  f"({rate:.2f} fps)  "
                  f"cumulative: -{total_n} nodes, -{total_e} edges "
                  f"in {touched} frames{extra}")
    if mission_vis_dir is not None:
        print(f"    visualizations saved to {mission_vis_dir}")
    return total_n, total_e, touched


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Depth-Anything-based occlusion cleaner for nav-graph nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mission-root", type=Path, required=True, metavar="DIR",
                    help="Parent directory whose subfolders are missions.")
    ap.add_argument("--missions", nargs="+", metavar="NAME",
                    help="Mission folder names or partial matches. Default: all.")
    ap.add_argument("--revert", action="store_true",
                    help="Move occlusion_cleaned_* back into nodes/edges and exit.")
    ap.add_argument("--model", default=OUTDOOR_MODEL,
                    help=f"Depth Anything model id. Default: {OUTDOOR_MODEL}")
    ap.add_argument("--indoor", action="store_true",
                    help=f"Use the indoor metric model ({INDOOR_MODEL}).")
    ap.add_argument("--window", type=int, default=7, metavar="N",
                    help="Side length of depth-sampling window in pixels. Default: 7.")
    ap.add_argument("--agg", choices=AGG_CHOICES, default="adaptive",
                    help="How to aggregate depth over the window. "
                         "'adaptive' (default) uses median for smooth "
                         "patches and p25 only when there's a real "
                         "foreground discontinuity in the window — avoids "
                         "false positives on close ground nodes while "
                         "still catching thin occluders. "
                         "min/p10/p25 always pick the lower tail (more "
                         "aggressive); median is conservative.")
    ap.add_argument("--calibrate", choices=CALIBRATE_CHOICES, default="inverse",
                    help="Per-frame correction of Depth Anything's "
                         "systematic depth error using the frame's nodes as "
                         "reference points (Theil-Sen, robust to outliers). "
                         "'inverse' fits in 1/depth space — handles the "
                         "common DA pattern where absolute error is large "
                         "at close range and small at far. 'linear' fits "
                         "scale+bias in metric depth (good if the error "
                         "is uniform). 'bias' is an additive offset only. "
                         "'none' disables. Default: inverse.")
    ap.add_argument("--abs-tol", type=float, default=0.5, metavar="M",
                    help="Absolute depth tolerance in metres. Default: 0.5.")
    ap.add_argument("--rel-tol", type=float, default=0.15, metavar="F",
                    help="Relative depth tolerance (fraction of pred depth). Default: 0.15.")
    ap.add_argument("--device", type=int, default=None, metavar="ID",
                    help="CUDA device id, -1 for CPU. Default: auto.")
    ap.add_argument("--vis-dir", type=Path, default=None, metavar="DIR",
                    help="If set, for every frame where occlusion removed any "
                         "nodes, save vis files to <vis-dir>/<mission_name>/. "
                         "Stale files from prior runs of each mission are "
                         "cleared first. By default both overlay and depth "
                         "vis are saved; use --no-overlay-vis / --no-depth-vis "
                         "to skip either.")
    ap.add_argument("--no-overlay-vis", action="store_true",
                    help="Skip saving overlay_*_occluded.png (the original "
                         "overlay with magenta X over removed nodes).")
    ap.add_argument("--no-depth-vis", action="store_true",
                    help="Skip saving depth_*_occluded.png (the colorized "
                         "predicted depth map with all tested nodes drawn).")
    args = ap.parse_args()

    if not args.mission_root.is_dir():
        sys.exit(f"[error] mission-root not a directory: {args.mission_root}")

    missions = get_missions(args.mission_root, args.missions)
    if not missions:
        sys.exit("[error] no missions found.")

    print(f"\n  Mission root : {args.mission_root}")
    print(f"  Missions     : {len(missions)}")
    for mp in missions:
        print(f"    {mp.name}")

    # ── Revert path ──
    if args.revert:
        gn = ge = gf = gr = 0
        for mp in missions:
            f, r, n, e = revert_mission(mp)
            gf += f; gr += r; gn += n; ge += e
            print(f"  [{mp.name}]  scanned {f}, restored {r} "
                  f"({n} nodes, {e} edges)")
        print(f"\n  Done.  Restored {gn} nodes / {ge} edges across "
              f"{gr}/{gf} files.")
        return

    # ── Forward (occlusion clean) path ──
    model_name = INDOOR_MODEL if args.indoor else args.model
    print(f"\n  Window       : {args.window}px ({args.agg})")
    print(f"  Calibration  : {args.calibrate}")
    print(f"  Tolerance    : max({args.abs_tol} m, {args.rel_tol} * pred)")
    if args.vis_dir is not None:
        args.vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Vis output   : {args.vis_dir}")
    est = DepthEstimator(model_name, device_arg=args.device)

    grand_n = grand_e = 0
    for mp in missions:
        n, e, _ = process_mission(mp, est, args.window, args.abs_tol,
                                  args.rel_tol, args.agg, args.calibrate,
                                  vis_dir=args.vis_dir,
                                  save_overlay=not args.no_overlay_vis,
                                  save_depth=not args.no_depth_vis)
        grand_n += n
        grand_e += e

    print(f"\n  Done.  Moved {grand_n} nodes / {grand_e} edges into "
          f"occlusion_cleaned_*.")
    print(f"  Re-run with --revert to undo.")


if __name__ == "__main__":
    main()
