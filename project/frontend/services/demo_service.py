"""demo_service.py — Business logic for demo chart and slice-comparison generation.

Blueprints call these functions; heavy imports (matplotlib, PIL, tifffile) live here,
not in route handlers.  All functions take plain Path arguments and raise on failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Cell-count chart
# ---------------------------------------------------------------------------

_CHART_CODE = r"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, sys
hier = pd.read_csv(sys.argv[1])
d2 = hier[hier['depth']==2].sort_values('count',ascending=False)
d2 = d2[d2['count']>0]
colors=['#E57373','#FF9800','#FFEB3B','#66BB6A','#26C6DA','#5C6BC0','#AB47BC','#EC407A','#26A69A','#8D6E63','#78909C','#FFA726']
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6)); fig.patch.set_facecolor('#0e0e16')
for ax in [ax1,ax2]:
    ax.set_facecolor('#141420'); ax.tick_params(colors='#cccccc',labelsize=9)
    ax.spines['bottom'].set_color('#444'); ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
bars=ax1.barh(range(len(d2)),d2['count'].values,color=[colors[i%len(colors)] for i in range(len(d2))],edgecolor='none',height=0.7)
ax1.set_yticks(range(len(d2))); ax1.set_yticklabels([f"{r['acronym']} ({r['region_name'][:20]})" for _,r in d2.iterrows()],fontsize=8.5,color='#cccccc')
ax1.set_xlabel('Cell Count',color='#cccccc',fontsize=10); ax1.set_title('Cell Counts by Major Brain Region',color='white',fontsize=12,pad=10)
[ax1.text(b.get_width()*1.01,b.get_y()+b.get_height()/2,f'{int(r.count):,}',va='center',ha='left',fontsize=8,color='#aaa') for b,r in zip(bars,d2.itertuples())]
total=d2['count'].sum(); pcts=d2['count'].values/total*100
labels=[f"{r['acronym']} {p:.1f}%" if p>3 else '' for (_,r),p in zip(d2.iterrows(),pcts)]
ax2.pie(d2['count'].values,colors=[colors[i%len(colors)] for i in range(len(d2))],labels=labels,labeldistance=1.1,textprops={'fontsize':8,'color':'#cccccc'},startangle=90,wedgeprops={'edgecolor':'#0e0e16','linewidth':1.5})
ax2.set_title('Proportion by Brain Region',color='white',fontsize=12,pad=10)
root=hier[hier['depth']==0]['count'].values[0]
fig.suptitle(f'Sample 35 — Whole Brain Cell Counts  |  Total: {int(root):,} cells  |  {len(hier)} regions mapped',color='white',fontsize=11,y=0.98)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(sys.argv[2],dpi=120,bbox_inches='tight',facecolor='#0e0e16',edgecolor='none')
"""


def generate_cell_chart(hier_path: Path, chart_path: Path, project_root: Path) -> None:
    """Generate cell-count bar+pie chart from *hier_path* CSV, save to *chart_path*.

    Raises ``subprocess.CalledProcessError`` on failure.
    """
    subprocess.run(
        [sys.executable, '-c', _CHART_CODE, str(hier_path), str(chart_path)],
        cwd=str(project_root),
        timeout=60,
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Slice comparison image
# ---------------------------------------------------------------------------

def generate_demo_comparison(
    slice_idx: int,
    reg_dir: Path,
    data_dir: Path,
    out_path: Path,
) -> None:
    """Generate side-by-side raw vs atlas comparison for *slice_idx*, save to *out_path*.

    Raises ``FileNotFoundError`` if the overlay PNG for *slice_idx* is missing,
    or any other exception encountered during image composition.
    """
    import numpy as np
    import tifffile as tf
    from PIL import Image, ImageDraw, ImageFont
    from scripts.make_demo_panel import _vibrant_recolor, _crop_to_brain

    ov_path = reg_dir / f'slice_{slice_idx:04d}_overlay.png'
    lbl_path = reg_dir / f'slice_{slice_idx:04d}_registered_label.tif'

    if not ov_path.exists():
        raise FileNotFoundError(f"overlay not found: {ov_path}")

    with Image.open(str(ov_path)) as _img:
        ov = np.array(_img.convert('RGB'))
    lbl = tf.imread(str(lbl_path)) if lbl_path.exists() else np.zeros(ov.shape[:2], dtype=np.int32)

    raw_files = sorted(data_dir.glob('*.tif'))
    raw_file = raw_files[min(slice_idx, len(raw_files) - 1)] if raw_files else None
    if raw_file:
        raw_orig = tf.imread(str(raw_file))
        p1, p99 = np.percentile(raw_orig[raw_orig > 0], [2, 98]) if raw_orig.max() > 0 else (0, 1)
        raw_norm = np.clip(
            (raw_orig.astype(np.float32) - p1) / (p99 - p1 + 1e-6) * 255, 0, 255
        ).astype(np.uint8)
        raw_rgb = np.stack([raw_norm] * 3, axis=-1)
    else:
        raw_rgb = np.zeros_like(ov)

    vibrant = _vibrant_recolor(ov, lbl)
    raw_crop = _crop_to_brain(raw_rgb, pad=25)
    vib_crop = _crop_to_brain(vibrant, pad=25)

    H = max(raw_crop.shape[0], vib_crop.shape[0])

    def _resize_h(arr: np.ndarray, h: int) -> np.ndarray:
        img = Image.fromarray(arr)
        sc = h / img.height
        return np.array(img.resize((int(img.width * sc), h), Image.LANCZOS))

    raw_r = _resize_h(raw_crop, H)
    vib_r = _resize_h(vib_crop, H)
    div = np.full((H, 6, 3), 35, dtype=np.uint8)
    combined = np.concatenate([raw_r, div, vib_r], axis=1)

    W = combined.shape[1]
    header = np.full((50, W, 3), 18, dtype=np.uint8)
    body = Image.fromarray(np.concatenate([header, combined], axis=0))
    draw = ImageDraw.Draw(body)
    try:
        font = ImageFont.truetype('arial.ttf', 22)
        sm = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = sm = ImageFont.load_default()

    draw.text((10, 14), 'Raw Lightsheet', fill=(200, 200, 200), font=font)
    draw.text((raw_r.shape[1] + 14, 14), 'Brainfast — Atlas Registration', fill=(200, 200, 200), font=font)
    draw.text((W - 260, 32), f'Slice {slice_idx} · Allen CCFv3', fill=(100, 100, 100), font=sm)
    body.save(str(out_path), quality=92)
