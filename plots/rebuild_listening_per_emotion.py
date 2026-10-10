"""Plot the existing per-emotion speaker/emotion identity tradeoffs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parent
OFFSETS = [
    [(0, 11), (5, 12), (-16, -10), (5, -12), (17, -2), (0, 15)],
    [(5, 10), (4, 11), (3, 11), (0, -12), (7, -12), (-13, 0)],
    [(0, 11), (4, -12), (-4, -12), (8, 10), (-13, -3), (3, 11)],
    [(0, 11), (5, 12), (-1, 13), (3, -12), (0, -13), (-4, -13)],
    [(0, 11), (5, 12), (0, 12), (7, -12), (0, -12), (-7, 12)],
    [(0, 11), (0, -12), (-6, -12), (7, -12), (8, 13), (-8, 13)],
]

MODELS = [('inferred','IEVC','#0072B2','o'),('native','NEVC','#D55E00','s')]

def main():
    source=ROOT/'supplement/tradeoff-curves-95ci.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    curves=pd.read_csv(source)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(7.15,5.1),dpi=160)
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.875,hspace=.43,wspace=.19)
    fig.legend(handles=[Line2D([0],[0],color=c,marker=mk,label=name,lw=1.5)
                        for _,name,c,mk in MODELS],loc='upper center',ncol=2,frameon=False)
    all_points=[];records=[]
    for row,condition in enumerate(['ssst','dsst']):
        for col,emotion in enumerate(['Angry','Happy','Sad']):
            ax=axes[row,col];points=[]
            for model,name,color,marker in MODELS:
                d=curves[(curves.condition==condition)&(curves.emotion==emotion)&
                         (curves.model_variant==model)].sort_values('alpha')
                assert len(d)==3 and np.allclose(d.alpha,[0,.6,1])
                assert d.eligible_pairs.eq(4).all()
                assert d.speaker_similarity_ci_status.eq('ok').all() and d.perceived_accuracy_ci_status.eq('ok').all()
                x=d.speaker_similarity;y=d.perceived_accuracy
                ax.errorbar(x,y,xerr=np.stack([x-d.speaker_similarity_ci_low,d.speaker_similarity_ci_high-x]),
                            yerr=np.stack([y-d.perceived_accuracy_ci_low,d.perceived_accuracy_ci_high-y]),
                            fmt='none',ecolor=color,alpha=.16,elinewidth=.75,capsize=1.8,zorder=1)
                ax.plot(x,y,color=color,marker=marker,ms=3.8,lw=1.4,zorder=4)
                points.extend((r.speaker_similarity,r.perceived_accuracy,r.alpha,color) for r in d.itertuples())
                records.append({'condition':condition,'emotion':emotion,'model':model,'alphas':d.alpha.tolist(),'pairs_per_point':4})
            ax.set_title(f'({chr(97+row*3+col)}) {condition.upper()} / {emotion}',fontsize=8.5)
            ax.set_xlim(.85,5.15);ax.set_ylim(-.08,1.08);ax.set_xticks([1,2,3,4,5]);ax.set_yticks([0,.25,.5,.75,1])
            ax.set_xlabel('Speaker identity (1–5)' if row==1 else '')
            ax.set_ylabel('Emotion identity: accuracy' if col==0 else '')
            if col:ax.tick_params(labelleft=False)
            ax.grid(alpha=.12,lw=.5)
            all_points.append((ax,points))
    fig.canvas.draw()
    labels=0
    for (ax,points),offsets in zip(all_points,OFFSETS):
        for (x,y,alpha,color),offset in zip(points,offsets):
            if not any(np.isclose(alpha,a) for a in [0,.6,1]):
                continue
            ax.annotate(f'{alpha:g}',(x,y),xytext=offset,textcoords='offset points',
                        ha='center',va='center',fontsize=7,color=color,zorder=8,
                        arrowprops=dict(arrowstyle='-',color=color,lw=.4,alpha=.7,shrinkA=2,shrinkB=4),
                        path_effects=[pe.withStroke(linewidth=2,foreground='white',alpha=.94)])
            labels+=1
    assert labels==36
    for ext in ['pdf','png','svg']:
        fig.savefig(ROOT/f'figures/listening-per-emotion-identity-tradeoffs.{ext}',dpi=240,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    (ROOT/'listening-per-emotion-audit.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),
        'source_sha256':digest,'curves':records,'alpha_labels':labels,'new_intervals':0,'new_experiments':0},indent=2)+'\n')
    print('Generated six panels, 12 curves and 36 alpha labels.')

if __name__=='__main__':
    main()
