from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent
d=pd.read_csv(P/'supplement/all-model-curves-with-95ci.csv')
models=['MODEL_inferred_intensity','MODEL_native_intensity_ranker','MODEL_native_intensity_speaker_cls','CycleGAN']
names=['IEVC','NEVC','NEVC+SC','CycleGAN'];colors=['#0072B2','#D55E00','#009E73','#CC79A7'];markers=['o','s','^','D']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
for setting,title in [('esd-ssst','ESD SSST'),('esd-dsst','ESD DSST'),('cremad','CREMA-D')]:
 sub=d[(d.setting==setting)&d.model.isin(models)&(d.emotion!='overall')];emotions=[e for e in ['Angry','Happy','Sad','Surprise'] if e in set(sub.emotion)]
 fig,axes=plt.subplots(2,2,figsize=(9.5,7.1),sharex=True,sharey=True)
 for ax,emo in zip(axes.flat,emotions):
  for j,m in enumerate(models):
   g=sub[(sub.emotion==emo)&(sub.model==m)].sort_values('alpha');assert len(g)==6
   x=g.emotion_ref_cosine_mean.to_numpy();y=g.speaker_cosine_mean.to_numpy()
   if m!='CycleGAN':ax.errorbar(x,y,xerr=g.emotion_ref_cosine_ci_half,yerr=g.speaker_cosine_ci_half,fmt='none',ecolor=colors[j],alpha=.13,elinewidth=3,capsize=0,zorder=1)
   ax.plot(x,y,color=colors[j],marker=markers[j],markersize=4,linewidth=1.6,linestyle='--' if m=='CycleGAN' else '-',zorder=3)
   for k,alpha in enumerate(g.alpha):
    if any(np.isclose(alpha,v) for v in [0,.6,1]):
     offsets=[(-13,10),(9,14),(-12,-15),(10,-13)];dx,dy=offsets[j]
     ax.annotate(f'{alpha:g}',(x[k],y[k]),xytext=(dx,dy),textcoords='offset points',fontsize=8,color=colors[j],bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.25),arrowprops=dict(arrowstyle='-',color=colors[j],lw=.45),zorder=4)
  ax.set_title(emo);ax.grid(alpha=.15);ax.margins(.17)
 for ax in list(axes.flat)[len(emotions):]:ax.set_visible(False)
 fig.supxlabel('Emotion-reference cosine');fig.supylabel('Source-speaker cosine');fig.suptitle(title,y=.99)
 fig.legend([Line2D([0],[0],color=c,marker=mk,ls='--' if n=='CycleGAN' else '-') for c,mk,n in zip(colors,markers,names)],names,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False)
 fig.tight_layout(rect=(.03,.03,1,.91))
 for ext in ['pdf','png']:fig.savefig(P/f'figures/appendix-{setting}-per-emotion.{ext}',dpi=180)
 plt.close(fig)
print('Generated three appendix figures from 264 existing mean curve points.')
l=pd.read_csv(P/'supplement/tradeoff-curves-95ci.csv')
fig,axes=plt.subplots(2,3,figsize=(10,5.7),sharex=True,sharey=True)
for i,condition in enumerate(['ssst','dsst']):
 for j,emotion in enumerate(['Angry','Happy','Sad']):
  ax=axes[i,j]
  for k,model in enumerate(['inferred','native']):
   g=l[(l.model_variant==model)&(l.condition==condition)&(l.emotion==emotion)].sort_values('alpha');assert len(g)==3
   y=g.emotion_strength.to_numpy();ax.errorbar(g.alpha,y,yerr=np.vstack([y-g.emotion_strength_ci_low,g.emotion_strength_ci_high-y]),fmt='none',ecolor=colors[k],alpha=.18,elinewidth=3,capsize=0)
   ax.plot(g.alpha,y,color=colors[k],marker=markers[k],label=names[k]);ax.set_title(f'{condition.upper()} / {emotion}');ax.set_xticks([0,.6,1]);ax.set_ylim(.9,5.1);ax.grid(alpha=.15)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
fig.supxlabel(r'Control coordinate $\alpha$');fig.supylabel('Rated target-emotion strength (1–5)');fig.tight_layout(rect=(.025,.02,1,.92))
for ext in ['pdf','png']:fig.savefig(P/f'figures/appendix-listening-alpha-strength.{ext}',dpi=180)
plt.close(fig)
