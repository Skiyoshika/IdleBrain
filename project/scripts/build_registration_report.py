from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return {row[0]: float(row[1]) for row in reader if len(row) >= 2}


def _rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _delta_text(
    after: float, before: float | None, *, lower_is_better: bool = False
) -> tuple[str, str]:
    if before is None:
        return "final only", "neutral"
    delta = float(after) - float(before)
    if abs(delta) < 1e-6:
        return "no change", "neutral"
    direction = -delta if lower_is_better else delta
    tone = "good" if direction > 0 else "bad"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}", tone


def _quick_verdict(metrics: dict[str, float], pre_metrics: dict[str, float]) -> tuple[str, str]:
    if not pre_metrics:
        return "Needs visual check", "Only final metrics are available."

    ncc_up = metrics.get("NCC", 0.0) >= pre_metrics.get("NCC", 0.0)
    ssim_up = metrics.get("SSIM", 0.0) >= pre_metrics.get("SSIM", 0.0)
    dice_ok = metrics.get("Dice", 0.0) >= (pre_metrics.get("Dice", 0.0) - 0.005)
    psnr_up = metrics.get("PSNR", 0.0) >= pre_metrics.get("PSNR", 0.0)
    mse_down = metrics.get("MSE", 1e9) <= pre_metrics.get("MSE", 1e9)

    good = sum([ncc_up, ssim_up, dice_ok, psnr_up, mse_down])
    if good >= 4:
        return "Looks improved", "Most quality metrics moved in the right direction."
    if good >= 3:
        return "Probably usable", "Metrics improved overall, but still check the picture manually."
    return "Needs review", "Metric changes are mixed. Inspect the before/after images."


def _metric_card_rows(metrics: dict[str, float], pre_metrics: dict[str, float]) -> str:
    rows: list[str] = []
    for key in ("NCC", "SSIM", "Dice", "MSE", "PSNR"):
        before = pre_metrics.get(key) if pre_metrics else None
        after = metrics.get(key, 0.0)
        delta_text, tone = _delta_text(after, before, lower_is_better=(key == "MSE"))
        before_text = f"{before:.4f}" if before is not None else "-"
        rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{before_text}</td>"
            f"<td>{after:.4f}</td>"
            f"<td class='delta {tone}'>{html.escape(delta_text)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _staining_card_rows(staining: dict[str, object]) -> str:
    fields = [
        ("Atlas Coverage", "atlas_coverage"),
        ("Staining Rate", "staining_rate"),
        ("Positive / Atlas", "positive_fraction_of_atlas"),
        ("Mean Signal", "mean_signal"),
        ("Threshold", "signal_threshold"),
    ]
    rows: list[str] = []
    for label, key in fields:
        try:
            value = float(staining.get(key, 0.0))
        except Exception:
            value = 0.0
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{value:.4f}</td></tr>")
    return "\n".join(rows)


def _friendly_file_links(run_dir: Path, metadata: dict[str, object]) -> str:
    items: list[tuple[str, str, Path | None]] = [
        ("Open Final Report", "This page is the main thing to check.", run_dir / "report.html"),
        ("Final Overview", "Quick visual summary after all refinement.", run_dir / "overview.png"),
        (
            "Before Overview",
            "Visual summary before Laplacian refinement.",
            run_dir / "overview_before.png",
        ),
        (
            "Final Volume",
            "Final registered brain NIfTI.",
            Path(str(metadata.get("registered_brain", "")))
            if metadata.get("registered_brain")
            else None,
        ),
        (
            "Before Laplacian",
            "The ANTs result before final refinement.",
            Path(str(metadata.get("registered_brain_pre_laplacian", "")))
            if metadata.get("registered_brain_pre_laplacian")
            else None,
        ),
        ("Metrics CSV", "Raw numbers for the final result.", run_dir / "registration_metrics.csv"),
        (
            "Metadata JSON",
            "Paths, parameters, and run details.",
            run_dir / "registration_metadata.json",
        ),
    ]
    cards: list[str] = []
    for title, desc, path in items:
        if path is None or not path.exists():
            continue
        href = _rel(path, run_dir)
        safe_href = html.escape(href)
        safe_title = html.escape(title)
        safe_desc = html.escape(desc)
        safe_name = html.escape(path.name)
        cards.append(
            f"<a class='file-card' href='{safe_href}'>"
            f"<strong>{safe_title}</strong>"
            f"<span>{safe_desc}</span>"
            f"<code>{safe_name}</code>"
            "</a>"
        )
    return "\n".join(cards)


def build_run_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    meta_path = run_dir / "registration_metadata.json"
    metrics_path = run_dir / "registration_metrics.csv"
    if not meta_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"missing registration outputs under {run_dir}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = _load_metrics(metrics_path)
    pre_metrics = metadata.get("metrics_before_laplacian", {}) or {}
    staining = metadata.get("staining_stats", {}) or {}
    annotation_path = (
        Path(str(metadata.get("annotation_fixed_half", "")))
        if metadata.get("annotation_fixed_half")
        else None
    )
    pre_path = (
        Path(str(metadata.get("registered_brain_pre_laplacian", "")))
        if metadata.get("registered_brain_pre_laplacian")
        else None
    )

    if pre_path and pre_path.exists() and annotation_path and annotation_path.exists():
        before_overview = run_dir / "overview_before.png"
        if not before_overview.exists():
            from scripts.make_registration_overview import make_registration_overview

            make_registration_overview(
                pre_path,
                annotation_path,
                before_overview,
                slices=[200, 300],
                structure_csv=PROJECT_DIR / "configs" / "allen_mouse_structure_graph.csv",
            )

    verdict_title, verdict_body = _quick_verdict(metrics, pre_metrics)
    report_path = run_dir / "report.html"
    overview_after = run_dir / "overview.png"
    overview_before = run_dir / "overview_before.png"
    before_img_html = ""
    if overview_before.exists():
        before_img_html = (
            "<figure><img src='overview_before.png' alt='Before refinement overview'>"
            "<figcaption>Before Laplacian refinement</figcaption></figure>"
        )
    after_img_html = ""
    if overview_after.exists():
        after_img_html = (
            "<figure><img src='overview.png' alt='Final overview'>"
            "<figcaption>Final result</figcaption></figure>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Brainfast Report</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffaf2;
      --ink: #17211d;
      --muted: #55635d;
      --line: #d7c8af;
      --good: #1f7a52;
      --bad: #b44a3f;
      --accent: #b9752f;
      --accent-soft: #f4dfc4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
      background: linear-gradient(180deg, #efe7d8 0%, var(--bg) 42%, #fbf7ef 100%);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 40px; }}
    .hero {{
      background: radial-gradient(circle at top left, #fff6e8 0%, var(--panel) 58%, #f6ead8 100%);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 24px;
      box-shadow: 0 12px 34px rgba(85, 56, 12, 0.08);
    }}
    .eyebrow {{ color: var(--accent); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; font-size: 12px; }}
    h1 {{ margin: 10px 0 8px; font-size: 32px; line-height: 1.15; }}
    .sub {{ color: var(--muted); font-size: 16px; max-width: 880px; }}
    .verdict {{
      margin-top: 18px;
      display: inline-block;
      padding: 12px 16px;
      border-radius: 14px;
      background: var(--accent-soft);
      border: 1px solid var(--line);
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 24px rgba(85, 56, 12, 0.05);
    }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    p, li {{ color: var(--muted); line-height: 1.55; }}
    ul {{ margin: 0; padding-left: 18px; }}
    .compare {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    figure {{
      margin: 0;
      background: #201712;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid #3a2f27;
    }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{
      padding: 10px 12px;
      background: #2c211a;
      color: #f4ede5;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{ color: var(--muted); font-weight: 700; }}
    .delta.good {{ color: var(--good); font-weight: 700; }}
    .delta.bad {{ color: var(--bad); font-weight: 700; }}
    .delta.neutral {{ color: var(--muted); }}
    .files {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .file-card {{
      display: block;
      text-decoration: none;
      color: inherit;
      background: #fffef9;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .file-card strong {{ display: block; margin-bottom: 6px; }}
    .file-card span {{ display: block; color: var(--muted); font-size: 14px; margin-bottom: 8px; }}
    .file-card code {{ color: var(--accent); }}
    .meta {{
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 8px 12px;
      font-size: 14px;
      margin-top: 8px;
    }}
    .meta strong {{ color: var(--ink); }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Brainfast Report</div>
      <h1>{html.escape(Path(str(metadata.get("input_source", ""))).name)}</h1>
      <p class="sub">Open this page first. If the two images below look aligned and the metric deltas are mostly green, this run is probably good enough for internal use.</p>
      <div class="verdict"><strong>{html.escape(verdict_title)}</strong><br>{html.escape(verdict_body)}</div>
      <div class="meta">
        <strong>Pipeline</strong><span>{html.escape(str(metadata.get("backend", "")).upper())}{" + Laplacian" if metadata.get("laplacian_enabled") else ""}</span>
        <strong>Hemisphere</strong><span>{html.escape(str(metadata.get("hemisphere", "")))}</span>
        <strong>Target Resolution</strong><span>{html.escape(str(metadata.get("target_um", "")))} um</span>
        <strong>Run Folder</strong><span>{html.escape(str(run_dir))}</span>
      </div>
    </section>

    <section class="compare">
      {before_img_html}
      {after_img_html}
    </section>

    <section class="grid">
      <article class="card">
        <h2>How To Check</h2>
        <ul>
          <li>First look at the two overview images above.</li>
          <li>The brain shape should sit inside the colored atlas regions without obvious large offsets.</li>
          <li>If the final image looks cleaner than the before image and the table below is mostly green, accept the run.</li>
        </ul>
      </article>
      <article class="card">
        <h2>Metric Summary</h2>
        <table>
          <thead>
            <tr><th>Metric</th><th>Before</th><th>Final</th><th>Change</th></tr>
          </thead>
          <tbody>
            {_metric_card_rows(metrics, pre_metrics)}
          </tbody>
        </table>
      </article>
      <article class="card">
        <h2>Staining Summary</h2>
        <p>Atlas Coverage tells you how much of the atlas half-brain contains signal. Staining Rate tells you how much of that covered tissue is actually bright enough to count as stained.</p>
        <table>
          <tbody>
            {_staining_card_rows(staining)}
          </tbody>
        </table>
      </article>
    </section>

    <section class="card" style="margin-top: 18px;">
      <h2>Important Files</h2>
      <div class="files">
        {_friendly_file_links(run_dir, metadata)}
      </div>
      <div class="footer">You do not need to inspect every file. Start with this page, then click the final overview or metrics only if something looks wrong.</div>
    </section>
  </div>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")

    (run_dir / "OPEN_ME_FIRST.txt").write_text(
        "Open report.html first.\n\n"
        "If the final image looks better than the before image and the metric changes are mostly green, the run is probably usable.\n",
        encoding="utf-8",
    )
    return report_path


def build_outputs_index(outputs_root: Path) -> Path:
    outputs_root = outputs_root.resolve()
    rows: list[tuple[str, str, str, Path]] = []
    for child in sorted(outputs_root.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "registration_metadata.json"
        metrics_path = child / "registration_metrics.csv"
        report_path = child / "report.html"
        if not meta_path.exists() or not metrics_path.exists():
            continue
        if not report_path.exists():
            try:
                build_run_report(child)
            except Exception:
                continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = _load_metrics(metrics_path)
        verdict_title, _ = _quick_verdict(metrics, meta.get("metrics_before_laplacian", {}) or {})
        rows.append(
            (
                Path(str(meta.get("input_source", ""))).name,
                str(meta.get("backend", "")),
                verdict_title,
                report_path,
            )
        )

    cards = []
    for input_name, backend, verdict, report_path in rows:
        safe_href = html.escape(_rel(report_path, outputs_root))
        safe_title = html.escape(input_name)
        safe_backend = html.escape(backend.upper())
        safe_verdict = html.escape(verdict)
        safe_folder = html.escape(report_path.parent.name)
        cards.append(
            f"<a class='run-card' href='{safe_href}'>"
            f"<strong>{safe_title}</strong>"
            f"<span>{safe_backend}</span>"
            f"<em>{safe_verdict}</em>"
            f"<code>{safe_folder}</code>"
            "</a>"
        )

    index_path = outputs_root / "index.html"
    cards_html = "\n".join(cards)
    index_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Brainfast Outputs</title>
  <style>
    body { margin: 0; font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; background: #f4efe5; color: #17211d; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 28px 20px 40px; }
    .hero { background: #fffaf2; border: 1px solid #d7c8af; border-radius: 22px; padding: 22px; }
    h1 { margin: 0 0 8px; }
    p { color: #55635d; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-top: 18px; }
    .run-card { display: block; text-decoration: none; color: inherit; background: #fffef9; border: 1px solid #d7c8af; border-radius: 16px; padding: 16px; }
    .run-card strong, .run-card span, .run-card em, .run-card code { display: block; }
    .run-card span { color: #b9752f; margin-top: 6px; font-weight: 700; }
    .run-card em { color: #1f7a52; margin-top: 8px; font-style: normal; }
    .run-card code { color: #55635d; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Brainfast Output Index</h1>
      <p>Open one run card below. Each run has its own user-friendly report page.</p>
    </section>
    <section class="grid">
      __CARDS__
    </section>
  </div>
</body>
</html>
""".replace("__CARDS__", cards_html)
    index_path.write_text(index_html, encoding="utf-8")
    (outputs_root / "OPEN_ME_FIRST.txt").write_text(
        "Open index.html first.\n\nEach run folder also contains its own report.html.\n",
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build user-friendly HTML reports for Brainfast outputs"
    )
    ap.add_argument("--run-dir", default=None, help="Single run directory to build")
    ap.add_argument("--outputs-root", default="outputs", help="Outputs root for the index page")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    if not outputs_root.is_absolute():
        outputs_root = (PROJECT_DIR / outputs_root).resolve()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (PROJECT_DIR / run_dir).resolve()
        report = build_run_report(run_dir)
        print(f"Built run report -> {report}")

    index = build_outputs_index(outputs_root)
    print(f"Built outputs index -> {index}")


if __name__ == "__main__":
    main()
