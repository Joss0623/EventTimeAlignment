import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/home_new/qsmx/runs/AttentionPIT/alert01');r=json.load(open(R/'quiet_policy_secondary.json'))
keys=['Chronos2','PatchTST','iTransformer','TimeMixer','Numeric','Metadata','TFIDF','Qwen3','Mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct'];labels=['Chronos-2','PatchTST','iTransformer','TimeMixer','Numeric MLP','Metadata','TF-IDF','Qwen3','Mistral','gpt-oss','DeepSeek','Kimi'];colors=['#8b96a3']*5+['#98b7a3']*2+['#78a0b6']*5
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.size':2.5,'ytick.major.size':0,'pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(2,1,figsize=(3.35,4.05),gridspec_kw={'height_ratios':[2.15,1.15]});fig.subplots_adjust(left=.29,right=.96,top=.94,bottom=.1,hspace=.65)
y=np.arange(len(keys));v=[r['models'][k]['event_recall_percent'] for k in keys];ax.barh(y,v,color=colors,height=.64);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,min(103,max(v)+16));ax.set_xlabel('Event recall (%)');ax.set_title('(a) Quiet-history warning policy',loc='left',fontweight='bold',fontsize=8,pad=8)
for i,z in enumerate(v):ax.text(z+.7,i,f'{z:.1f}',va='center',fontsize=6.5)
ax.set_axisbelow(True);ax.grid(axis='x',color='#eceef0',linewidth=.5)
llms=keys[-5:];z=[r['paired_gains'][k]['event_recall_pp'] for k in llms];lo=[r['paired_gains'][k]['recall_CI95_pp'][0] for k in llms];hi=[r['paired_gains'][k]['recall_CI95_pp'][1] for k in llms];bx.axvline(0,color='#aeb3b8',ls='--',lw=.7);bx.errorbar(z,np.arange(5),xerr=[np.array(z)-lo,np.array(hi)-z],fmt='o',color='#78a0b6',ms=3.5,lw=.8,capsize=2);bx.set_yticks(np.arange(5),labels[-5:]);bx.invert_yaxis();bx.set_xlabel('Recall gain (pp), paired 95% CI');bx.set_title('(b) Relative to '+r['reference'],loc='left',fontweight='bold',fontsize=8,pad=8);fig.savefig(R/'alert_results.pdf');fig.savefig(R/'alert_results.png',dpi=240);plt.close(fig)
print('single-column figure complete')
