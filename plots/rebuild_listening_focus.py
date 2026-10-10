"""Two listening-study questions using the fixed, completed rating export."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
MODELS = [('inferred', 'IEVC', '#0072B2', 'o'), ('native', 'NEVC', '#D55E00', 's')]

def table(filename, caption, label, headers, rows):
    text = ('\\begin{table*}[t]\n\\centering\\scriptsize\n\\caption{' + caption + '}\n'
            '\\label{' + label + '}\n\\setlength{\\tabcolsep}{5pt}\n'
            '\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{|' + 'l'*len(headers) + '|}\n\\hline\n'
            + ' & '.join(headers) + ' \\\\\n\\hline\n'
            + '\n'.join(' & '.join(row) + ' \\\\' for row in rows)
            + '\n\\hline\n\\end{tabular}\n\\end{table*}\n')
    (ROOT / 'tables' / filename).write_text(text)

def interval(row, name):
    return f'{row[name]:.3f} [{row[name+"_ci_low"]:.3f}, {row[name+"_ci_high"]:.3f}]'

def main():
    paths = [ROOT/'supplement/listening-clip-means.csv', ROOT/'supplement/tradeoff-curves-95ci.csv']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    clips = pd.read_csv(paths[0]); curves = pd.read_csv(paths[1]); curves = curves[curves.emotion=='overall']
    assert len(clips)==144 and not clips.duplicated(['model_variant','condition','pair_family','alpha']).any()
    assert clips.emotion_ratings.sum()==366 and clips.speaker_ratings.sum()==396
    stats=[]; correlation_rows=[]; identity_rows=[]
    for condition in ['ssst', 'dsst']:
        for model,name,_,_ in MODELS:
            c=clips[(clips.model_variant==model)&(clips.condition==condition)]
            d=curves[(curves.model_variant==model)&(curves.condition==condition)].sort_values('alpha')
            assert len(c)==36 and c.pair_family.nunique()==12 and np.allclose(d.alpha,[0,.6,1])
            for row in d.to_dict('records'):
                values=c[np.isclose(c.alpha,row['alpha'])]
                assert len(values)==12
                for metric in ['emotion_strength','speaker_similarity','perceived_accuracy']:
                    assert np.isclose(values[metric].mean(),row[metric],atol=1e-12)
                identity_rows.append([condition.upper(),name,f'{row["alpha"]:g}',
                                      interval(row,'speaker_similarity'),interval(row,'perceived_accuracy')])
            # Spearman rho uses average ranks for tied alpha and rating means.
            rho=c.alpha.rank(method='average').corr(c.emotion_strength.rank(method='average'))
            stats.append(dict(model_variant=model,condition=condition,clips=len(c),pair_families=12,
                              spearman_alpha_strength=rho))
            correlation_rows.append([condition.upper(),name,'36',f'{rho:.3f}']+
                                    [f'{x:.3f}' for x in d.emotion_strength])
    pd.DataFrame(stats).to_csv(ROOT/'supplement/alpha-strength-correlations.csv',index=False)
    table('listening-alpha-strength.tex',
          r'Alpha and rated target-emotion strength. Spearman $\rho$ uses the 36 clip-level mean ratings '
          r'within each model/condition (12 pair families at three alpha values), with equal clip weights and average ranks for ties. '
          r'Mean strength is on the 1--5 scale. Correlations are descriptive; shared listeners and pair families are not independent replications.',
          'tab:alpha-strength',['Condition','Model','Clips',r'$\rho(\alpha,\mathrm{strength})$',
                               r'Mean at $\alpha=0$',r'Mean at $0.6$',r'Mean at $1$'],correlation_rows)
    table('listening-identity-tradeoff.tex',
          r'Speaker--emotion identity tradeoff across the study models and conditions. '
          r'Speaker identity is the mean 1--5 source-identity rating; emotion identity is target-category accuracy (0--1). '
          r'Each point averages the same 12 pair families. Brackets give the existing pointwise 95\% listener-cluster intervals, conditional on these stimuli.',
          'tab:listening-identity',['Condition','Model',r'$\alpha$',r'Speaker identity [95\% CI]',
                                  r'Emotion identity [95\% CI]'],identity_rows)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.8))
    for col,condition in enumerate(['ssst','dsst']):
        for model,name,color,marker in MODELS:
            d=curves[(curves.model_variant==model)&(curves.condition==condition)].sort_values('alpha')
            ax=axes[0,col]
            strength=d.emotion_strength
            ax.errorbar(d.alpha,strength,yerr=np.stack([strength-d.emotion_strength_ci_low,d.emotion_strength_ci_high-strength]),
                        fmt='none',ecolor=color,alpha=.18,elinewidth=.8,capsize=2,zorder=1)
            ax.plot(d.alpha,strength,color=color,marker=marker,ms=4,lw=1.5,zorder=3)
            ax=axes[1,col];x=d.speaker_similarity;y=d.perceived_accuracy
            ax.errorbar(x,y,xerr=np.stack([x-d.speaker_similarity_ci_low,d.speaker_similarity_ci_high-x]),
                        yerr=np.stack([y-d.perceived_accuracy_ci_low,d.perceived_accuracy_ci_high-y]),
                        fmt='none',ecolor=color,alpha=.18,elinewidth=.8,capsize=2,zorder=1)
            ax.plot(x,y,color=color,marker=marker,ms=4,lw=1.5,zorder=3)
            for row in d.itertuples():
                ax.annotate(f'{row.alpha:g}',(row.speaker_similarity,row.perceived_accuracy),
                            xytext=((-12,8) if model=='native' and row.alpha==1 else (5,6 if model=='inferred' else -11)),textcoords='offset points',
                            color=color,fontsize=7)
        axes[0,col].set_title(f'({chr(97+col)}) {condition.upper()}: alpha and strength',fontsize=9)
        axes[0,col].set_xlabel(r'Control $\alpha$');axes[0,col].set_xticks([0,.6,1]);axes[0,col].set_xlim(-.06,1.06)
        axes[0,col].set_ylim(1,5);axes[0,col].set_ylabel('Rated emotion strength (1–5)' if col==0 else '')
        axes[1,col].set_title(f'({chr(99+col)}) {condition.upper()}: identity tradeoff',fontsize=9)
        axes[1,col].set_xlabel('Perceived speaker identity (1–5)');axes[1,col].set_xlim(1,5.15)
        axes[1,col].set_ylim(0,1);axes[1,col].set_ylabel('Emotion identity: category accuracy' if col==0 else '')
        for ax in axes[:,col]:ax.grid(alpha=.14)
    handles=[Line2D([0],[0],color=c,marker=m,label=n,lw=1.5) for _,n,c,m in MODELS]
    fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.93),h_pad=1.4,w_pad=1.3)
    for ext in ['pdf','png']:
        fig.savefig(ROOT/f'figures/listening-control-and-identity.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    for p in paths:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))]
    (ROOT/'listening-focus-audit.json').write_text(json.dumps({'input_sha256':hashes,'clip_means_match_saved_curves':True,
        'correlations':stats,'new_experiments':0,'new_intervals':0,'snapshot':'3 October 2026; 762 completed ratings'},indent=2)+'\n')
    print(pd.DataFrame(stats).to_string(index=False))

if __name__=='__main__':
    main()
