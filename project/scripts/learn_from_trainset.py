from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import imread, imwrite
from skimage import morphology
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from scripts.atlas_autopick import autopick_best_z
from scripts.overlay_render import render_overlay


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    den = float(np.sum(aa) + np.sum(bb)) + 1e-6
    inter = float(np.sum(aa & bb))
    return float(2.0 * inter / den)


def _boundary_f1(pred: np.ndarray, target: np.ndarray, tol_px: int = 2) -> float:
    pp = pred.astype(bool)
    tt = target.astype(bool)
    if not np.any(pp) or not np.any(tt):
        return 0.0
    dil = morphology.disk(max(1, int(tol_px)))
    pp_d = morphology.dilation(pp.astype(np.uint8), dil).astype(bool)
    tt_d = morphology.dilation(tt.astype(np.uint8), dil).astype(bool)
    precision = float(np.sum(pp & tt_d)) / (float(np.sum(pp)) + 1e-6)
    recall = float(np.sum(tt & pp_d)) / (float(np.sum(tt)) + 1e-6)
    if precision + recall < 1e-8:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _match_shape_bool(a: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if tuple(a.shape) == tuple(target_shape):
        return a.astype(bool)
    arr = resize(
        a.astype(np.float32),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return (arr > 0.5)


def _safe_roll_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.roll(mask, shift=(int(dy), int(dx)), axis=(0, 1))
    if dy > 0:
        out[:dy, :] = False
    elif dy < 0:
        out[dy:, :] = False
    if dx > 0:
        out[:, :dx] = False
    elif dx < 0:
        out[:, dx:] = False
    return out


def _align_target_to_pred(target_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    tgt = _match_shape_bool(target_mask, pred_mask.shape).astype(bool)
    pp = pred_mask.astype(bool)
    if not np.any(pp) or not np.any(tgt):
        return tgt
    pyx = np.argwhere(pp)
    tyx = np.argwhere(tgt)
    py, px = np.mean(pyx[:, 0]), np.mean(pyx[:, 1])
    ty, tx = np.mean(tyx[:, 0]), np.mean(tyx[:, 1])
    dy = int(round(py - ty))
    dx = int(round(px - tx))
    return _safe_roll_mask(tgt, dy=dy, dx=dx)


def _extract_show_boundary_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.int16)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    cyan = (g > 90) & (b > 100) & (r < 120) & (((g + b) // 2 - r) > 28)
    bright = (np.maximum(np.maximum(r, g), b) > 155)
    sat = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15
    color_lines = cyan & bright & sat
    thin = morphology.opening(color_lines.astype(np.uint8), morphology.disk(1)).astype(bool)
    return thin


def _label_boundary_mask(label: np.ndarray) -> np.ndarray:
    lbl = label.astype(np.int32)
    outer = find_boundaries(lbl > 0, mode="outer", connectivity=2)
    inner = find_boundaries(lbl, mode="inner", connectivity=2)
    bd = (outer | inner)
    bd = morphology.dilation(bd.astype(np.uint8), morphology.disk(1)).astype(bool)
    return bd


def _pair_ids(train_dir: Path) -> list[str]:
    out = []
    for p in sorted(train_dir.glob("*_Ori.png")):
        sid = p.name.replace("_Ori.png", "")
        show = train_dir / f"{sid}_Show.png"
        if show.exists():
            out.append(sid)
    return out


def _png_to_tif(png_path: Path, tif_path: Path) -> None:
    with Image.open(png_path) as im:
        arr = np.array(im.convert("RGB"))
    # Use green channel as anatomical base for registration.
    g = arr[..., 1].astype(np.uint16)
    imwrite(str(tif_path), g)


@dataclass(frozen=True)
class Cfg:
    fit_mode: str
    edge_smooth_iter: int
    profile_name: str
    warp_params: dict

    def key(self) -> str:
        return f"fit={self.fit_mode}|smooth={self.edge_smooth_iter}|profile={self.profile_name}"


def _default_profiles() -> dict[str, dict]:
    return {
        "balanced": {
            "edge_inner_weight": 1.05,
            "edge_outer_weight": 0.52,
            "outside_penalty": 2.35,
            "mask_dice_weight": 0.34,
            "liquify_max_match_dist_ratio": 0.048,
            "liquify_max_disp_ratio": 0.065,
            "liquify_sigma_ratio": 0.018,
            "liquify_tps_ctrl": 260,
            "liquify_tps_smooth": 2.2,
            "liquify_prefer_margin": 0.0025,
            "liquify_prefer_min_gain": 0.005,
        },
        "internal_strong": {
            "edge_inner_weight": 1.18,
            "edge_outer_weight": 0.46,
            "outside_penalty": 2.55,
            "mask_dice_weight": 0.31,
            "liquify_max_match_dist_ratio": 0.055,
            "liquify_max_disp_ratio": 0.075,
            "liquify_sigma_ratio": 0.016,
            "liquify_tps_ctrl": 300,
            "liquify_tps_smooth": 1.8,
            "liquify_prefer_margin": 0.0020,
            "liquify_prefer_min_gain": 0.004,
        },
        "conservative": {
            "edge_inner_weight": 1.08,
            "edge_outer_weight": 0.55,
            "outside_penalty": 2.75,
            "mask_dice_weight": 0.36,
            "liquify_max_match_dist_ratio": 0.040,
            "liquify_max_disp_ratio": 0.050,
            "liquify_sigma_ratio": 0.022,
            "liquify_tps_ctrl": 220,
            "liquify_tps_smooth": 2.8,
            "liquify_prefer_margin": 0.0020,
            "liquify_prefer_min_gain": 0.006,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune registration params from train_data_set Ori/Show pairs")
    ap.add_argument("--train-dir", default=str(Path(__file__).resolve().parent.parent / "train_data_set"))
    ap.add_argument("--annotation", default=str(Path(__file__).resolve().parent.parent / "annotation_25.nii.gz"))
    ap.add_argument("--out-json", default=str(Path(__file__).resolve().parent.parent / "outputs" / "trainset_tuned_params.json"))
    ap.add_argument("--pixel-size-um", type=float, default=0.65)
    ap.add_argument("--major-top-k", type=int, default=48)
    ap.add_argument("--fit-modes", default="cover,contain", help="Comma-separated fit modes")
    ap.add_argument("--smooth-values", default="0,1", help="Comma-separated edge smooth iterations")
    ap.add_argument(
        "--profiles",
        default="balanced,internal_strong",
        help="Comma-separated profile names to search",
    )
    ap.add_argument("--sample-limit", type=int, default=0, help="Use only first N samples (0=all)")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    annotation = Path(args.annotation)
    out_json = Path(args.out_json)
    work_dir = out_json.parent / "_train_tune_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    ids = _pair_ids(train_dir)
    if not ids:
        raise RuntimeError(f"no *_Ori.png/*_Show.png pairs found in {train_dir}")
    if int(args.sample_limit) > 0:
        ids = ids[: int(args.sample_limit)]

    profiles = _default_profiles()
    fit_modes = [x.strip() for x in str(args.fit_modes).split(",") if x.strip()]
    smooth_values = [int(x.strip()) for x in str(args.smooth_values).split(",") if x.strip()]
    profile_names = [x.strip() for x in str(args.profiles).split(",") if x.strip()]
    profile_names = [p for p in profile_names if p in profiles]
    if not fit_modes:
        fit_modes = ["cover", "contain"]
    if not smooth_values:
        smooth_values = [0, 1]
    if not profile_names:
        profile_names = ["balanced", "internal_strong"]

    cfgs: list[Cfg] = []
    for fit_mode in fit_modes:
        for smooth in smooth_values:
            for profile_name in profile_names:
                wp = profiles[profile_name]
                cfgs.append(
                    Cfg(
                        fit_mode=fit_mode,
                        edge_smooth_iter=int(smooth),
                        profile_name=profile_name,
                        warp_params=dict(wp),
                    )
                )

    score_sum = {c.key(): 0.0 for c in cfgs}
    score_cnt = {c.key(): 0 for c in cfgs}
    per_sample: dict[str, dict] = {}

    for sid in ids:
        ori_png = train_dir / f"{sid}_Ori.png"
        show_png = train_dir / f"{sid}_Show.png"
        real_tif = work_dir / f"{sid}_ori_gray.tif"
        label_tif = work_dir / f"{sid}_auto_label.tif"
        _png_to_tif(ori_png, real_tif)

        auto_meta = autopick_best_z(
            real_path=real_tif,
            annotation_nii=annotation,
            out_label_tif=label_tif,
            z_step=1,
            pixel_size_um=float(args.pixel_size_um),
            slicing_plane="coronal",
            roi_mode="auto",
        )

        show_arr = np.array(Image.open(show_png).convert("RGB"))
        tgt_mask_raw = _extract_show_boundary_mask(show_arr)

        per_cfg: dict[str, float] = {}
        for cfg in cfgs:
            k = cfg.key()
            out_png = work_dir / f"{sid}_{k.replace('|', '_').replace('=', '-')}.png"
            warped_tif = work_dir / f"{sid}_{k.replace('|', '_').replace('=', '-')}_warped.tif"
            _, _diag = render_overlay(
                real_slice_path=real_tif,
                label_slice_path=label_tif,
                out_png=out_png,
                alpha=0.72,
                mode="contour-major",
                pixel_size_um=float(args.pixel_size_um),
                fit_mode=cfg.fit_mode,
                edge_smooth_iter=cfg.edge_smooth_iter,
                major_top_k=int(args.major_top_k),
                return_meta=True,
                warped_label_out=warped_tif,
                warp_params=cfg.warp_params,
            )
            warped = imread(str(warped_tif)).astype(np.int32)
            pred_mask = _label_boundary_mask(warped)
            tgt_mask = _align_target_to_pred(tgt_mask_raw, pred_mask)

            d = _dice(pred_mask, tgt_mask)
            f1 = _boundary_f1(pred_mask, tgt_mask, tol_px=2)
            # emphasize boundary overlap while keeping area agreement
            s = float(0.72 * f1 + 0.28 * d)
            per_cfg[k] = s
            score_sum[k] += s
            score_cnt[k] += 1

        best_k, best_s = max(per_cfg.items(), key=lambda x: x[1])
        best_cfg = next(c for c in cfgs if c.key() == best_k)
        per_sample[sid] = {
            "autopick": auto_meta,
            "scores": per_cfg,
            "best": {
                "key": best_k,
                "score": float(best_s),
                "fitMode": best_cfg.fit_mode,
                "edgeSmoothIter": int(best_cfg.edge_smooth_iter),
                "warpParams": dict(best_cfg.warp_params),
            },
        }

    mean_scores = {
        k: (score_sum[k] / max(score_cnt[k], 1))
        for k in score_sum
    }
    best_key, best_score = max(mean_scores.items(), key=lambda x: x[1])
    best_cfg = next(c for c in cfgs if c.key() == best_key)

    tuned = {
        "fitMode": best_cfg.fit_mode,
        "edgeSmoothIter": int(best_cfg.edge_smooth_iter),
        "warpParams": dict(best_cfg.warp_params),
    }

    out = {
        "train_dir": str(train_dir),
        "annotation": str(annotation),
        "n_samples": int(len(ids)),
        "major_top_k": int(args.major_top_k),
        "fit_modes": fit_modes,
        "smooth_values": smooth_values,
        "profile_names": profile_names,
        "metric": "0.72*boundary_f1(tol=2) + 0.28*dice",
        "mean_scores": mean_scores,
        "best": {"key": best_key, "score": float(best_score), "params": tuned},
        "profiles": profiles,
        "per_sample": per_sample,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(str(out_json))
    print(json.dumps(out["best"], ensure_ascii=False))


if __name__ == "__main__":
    main()
