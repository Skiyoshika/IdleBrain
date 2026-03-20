from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return {row[0]: float(row[1]) for row in reader if len(row) >= 2}


def _load_run(path: Path) -> dict[str, object] | None:
    meta_path = path / "registration_metadata.json"
    metrics_path = path / "registration_metrics.csv"
    if not meta_path.exists() or not metrics_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = _load_metrics(metrics_path)
    input_source = Path(meta.get("input_source", "")).name
    pre_metrics = meta.get("metrics_before_laplacian", {}) or {}
    lap = (meta.get("backend_details", {}) or {}).get("laplacian", {}) or {}
    return {
        "run_dir": str(path.resolve()),
        "input_file": input_source,
        "backend": meta.get("backend", ""),
        "laplacian_enabled": bool(meta.get("laplacian_enabled", False)),
        "target_um": meta.get("target_um", ""),
        "pad_z": meta.get("pad_z", ""),
        "pad_y": meta.get("pad_y", ""),
        "pad_x": meta.get("pad_x", ""),
        "normalize_input": meta.get("normalize_input", ""),
        "pre_NCC": pre_metrics.get("NCC", ""),
        "pre_SSIM": pre_metrics.get("SSIM", ""),
        "pre_Dice": pre_metrics.get("Dice", ""),
        "pre_MSE": pre_metrics.get("MSE", ""),
        "pre_PSNR": pre_metrics.get("PSNR", ""),
        "delta_NCC": (metrics.get("NCC", 0.0) - float(pre_metrics.get("NCC", 0.0)))
        if pre_metrics
        else "",
        "delta_SSIM": (metrics.get("SSIM", 0.0) - float(pre_metrics.get("SSIM", 0.0)))
        if pre_metrics
        else "",
        "delta_Dice": (metrics.get("Dice", 0.0) - float(pre_metrics.get("Dice", 0.0)))
        if pre_metrics
        else "",
        "delta_PSNR": (metrics.get("PSNR", 0.0) - float(pre_metrics.get("PSNR", 0.0)))
        if pre_metrics
        else "",
        "laplacian_points": lap.get("correspondences", ""),
        "laplacian_unique_voxels": lap.get("unique_boundary_voxels", ""),
        "laplacian_seconds": lap.get("solve_seconds", ""),
        **metrics,
    }


def collect_runs(outputs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for child in sorted(outputs_root.iterdir()):
        if not child.is_dir():
            continue
        row = _load_run(child)
        if row is not None:
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Brainfast 3D registration runs")
    ap.add_argument(
        "--outputs-root", default="outputs", help="Directory that contains run subdirectories"
    )
    ap.add_argument(
        "--out-csv",
        default="outputs/registration_benchmark.csv",
        help="Where to write the summary CSV",
    )
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    rows = collect_runs(outputs_root)
    if not rows:
        raise SystemExit(f"No registration runs found under {outputs_root}")

    fieldnames = [
        "input_file",
        "backend",
        "laplacian_enabled",
        "target_um",
        "pad_z",
        "pad_y",
        "pad_x",
        "normalize_input",
        "pre_NCC",
        "NCC",
        "delta_NCC",
        "pre_SSIM",
        "SSIM",
        "delta_SSIM",
        "pre_Dice",
        "Dice",
        "delta_Dice",
        "pre_MSE",
        "MSE",
        "pre_PSNR",
        "PSNR",
        "delta_PSNR",
        "laplacian_points",
        "laplacian_unique_voxels",
        "laplacian_seconds",
        "run_dir",
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r["input_file"]),
            str(r["backend"]),
            str(r.get("laplacian_enabled", False)),
            -float(r.get("NCC", 0.0)),
        ),
    )

    def _as_float(value: object) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    print(f"Wrote {len(rows_sorted)} run(s) -> {out_csv}")
    for row in rows_sorted:
        print(
            f"{row['input_file']} | {row['backend']} | "
            f"lap={row.get('laplacian_enabled', False)} | "
            f"NCC={_as_float(row.get('NCC', 0.0)):.4f} | "
            f"dNCC={_as_float(row.get('delta_NCC', 0.0)):.4f} | "
            f"SSIM={_as_float(row.get('SSIM', 0.0)):.4f} | "
            f"Dice={_as_float(row.get('Dice', 0.0)):.4f} | "
            f"PSNR={_as_float(row.get('PSNR', 0.0)):.4f}"
        )


if __name__ == "__main__":
    main()
