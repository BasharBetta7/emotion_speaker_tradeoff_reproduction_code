"""Plot existing all-model emotion-reference curves; no new measurements."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent
MODELS = ['MODEL_inferred_intensity', 'MODEL_native_intensity_ranker',
          'MODEL_native_intensity_speaker_cls', 'CycleGAN']
NAMES = ['IEVC', 'NEVC', 'NEVC+SC', 'CycleGAN']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#8E44AD']
MARKERS = ['o', 's', '^', 'D']
SETTINGS = [('esd-ssst', 'ESD SSST'), ('esd-dsst', 'ESD DSST'), ('cremad', 'CREMA-D')]
METRIC = 'emotion_ref_cosine'

def label_points(fig, ax, points):
    """Label alpha 0, 0.6 and 1, avoiding all markers and other labels."""
    renderer = fig.canvas.get_renderer()
    centers = np.array([ax.transData.transform((p[0], p[1])) for p in points])
    placed = []
    bounds = ax.get_window_extent().padded(-2)
    scale = fig.dpi / 72
    for index, (x, y, alpha, color) in enumerate(points):
        if not any(np.isclose(alpha, a) for a in [0, .6, 1]):
            continue
        text = f'{alpha:g}'
        width, height, _ = renderer.get_text_width_height_descent(
            text, matplotlib.font_manager.FontProperties(size=6.7), ismath=False)
        width += 3; height += 4
        origin = centers[index]
        candidates = []
        for radius in [12, 17, 23, 30, 38, 46]:
            for angle in np.arange(0, 360, 30):
                displacement = radius * scale * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                center = origin + displacement
                box = Bbox.from_bounds(center[0]-width/2, center[1]-height/2, width, height)
                outside = not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1))
                overlap = 0.0
                for prior in placed:
                    intersection = Bbox.intersection(box.padded(5), prior)
                    if intersection is not None:
                        overlap += intersection.width * intersection.height
                marker_hits = sum(box.padded(5).contains(*v) for v in centers)
                score = 1e7*outside + 1000*overlap + 10000*marker_hits + radius + angle/360
                candidates.append((score, displacement, box))
        _, displacement, box = min(candidates, key=lambda c: c[0])
        placed.append(box)
        offset = displacement / scale
        arrow = dict(arrowstyle='-', color=color, lw=.35, alpha=.65,
                     shrinkA=1, shrinkB=3) if np.linalg.norm(offset) > 9 else None
        ax.annotate(text, (x, y), xytext=offset, textcoords='offset points',
                    ha='center', va='center', fontsize=6.7, color=color,
                    arrowprops=arrow, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=.94)])
    return len(placed)

def main():
    source = ROOT / 'supplement/all-model-curves-with-95ci.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = pd.read_csv(source)
    data = data[data.emotion == 'overall'].copy()
    plt.rcParams.update({'font.size':8, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.15), dpi=160)
    fig.subplots_adjust(left=.075, right=.99, bottom=.15, top=.80, wspace=.30)
    handles = [Line2D([0], [0], color=c, marker=m, ms=4, lw=1.5,
                      ls='--' if i>=3 else '-', label=n)
               for i,(c,m,n) in enumerate(zip(COLORS, MARKERS, NAMES))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53,.994),
               ncol=4, frameon=False, fontsize=8, columnspacing=1.6)
    point_lists = []
    audit = []
    for panel, (ax, (setting, title)) in enumerate(zip(axes, SETTINGS)):
        points = []
        for i, model in enumerate(MODELS):
            d = data[(data.setting == setting) & (data.model == model)].sort_values('alpha')
            assert not d.empty
            assert len(d) == 6 and np.allclose(d.alpha, np.linspace(0,1,6))
            x, y = d.speaker_cosine_mean, d[METRIC+'_mean']
            if i < 3:
                assert (d.n_runs == 3).all()
                for opacity, width, caps in [(.035, 3.2, 0), (.14, .65, 1.5)]:
                    ax.errorbar(x, y, xerr=d.speaker_cosine_ci_half,
                                yerr=d[METRIC+'_ci_half'], fmt='none',
                                ecolor=COLORS[i], alpha=opacity, elinewidth=width,
                                capsize=caps, capthick=.55, zorder=1)
            else:
                assert d[METRIC+'_ci_half'].isna().all()
            ax.plot(x, y, color=COLORS[i], marker=MARKERS[i], ms=3.8,
                    lw=1.5, ls='--' if i>=3 else '-', zorder=4)
            points.extend((r.speaker_cosine_mean, getattr(r, METRIC+'_mean'),
                           r.alpha, COLORS[i]) for r in d.itertuples())
            audit.append({'setting':setting, 'model':model, 'alpha_values':d.alpha.tolist(), 'alpha_labels':[0,0.6,1],
                          'has_seed_ci':i<3})
        ax.set_title(f'({chr(97+panel)}) {title}', loc='left', fontsize=9, pad=5)
        ax.set_ylabel('Emotion-reference cosine' if panel == 0 else '')
        ax.set_xlabel('Source-speaker cosine', labelpad=2)
        ax.grid(alpha=.12, linewidth=.5)
        ax.margins(x=.055, y=.14)
        point_lists.append(points)
    fig.canvas.draw()
    count = sum(label_points(fig, ax, pts) for ax,pts in zip(axes,point_lists))
    assert count == 36
    outputs=[]
    for ext in ['pdf','png','svg']:
        path = ROOT / f'figures/all-model-emotion-reference-frontiers.{ext}'
        fig.savefig(path, dpi=240, bbox_inches='tight', pad_inches=.03)
        outputs.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    (ROOT / 'tradeoff-figure-audit.json').write_text(json.dumps({
        'metric':METRIC, 'source':str(source.relative_to(ROOT)), 'source_sha256':digest,
        'curves':audit, 'alpha_labels':count, 'outputs':outputs, 'layout':'one row, three columns',
        'ci_style':'unchanged intervals; opacity 0.14 with faint 0.035 soft underlay; behind mean curves',
        'experiments_run':0,
    },indent=2)+'\n')
    print(f'Generated cosine-only comparison: {len(audit)} curves, {count} alpha labels.')

if __name__ == '__main__':
    main()
