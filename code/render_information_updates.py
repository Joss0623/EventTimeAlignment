import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
O=Path('/home/home_new/qsmx/runs/AttentionPIT/expanded01/neural_readouts');d=json.load(open(O/'refresh_attribution.json'))['groups']['event']['1'];keys=['Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B'];names=['Qwen3','Mistral','gpt-oss','DeepSeek','Kimi'];plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],'font.size':6.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig=plt.figure(figsize=(3.35,3.05));a=fig.add_axes([.23,.60,.73,.23]);fig.text(.03,.985,'(a) Same readout, four information states',va='top',fontsize=7,fontweight='bold');mat=np.array([[d[k][c] for c in ['WAPE_A7','WAPE_B_numeric_only_update','WAPE_C_metadata_update','WAPE_D_current']] for k in keys]);cm=LinearSegmentedColormap.from_list('error',['#82acbd','#fbf7f1','#cc8d78']);im=a.imshow(mat,cmap=cm,vmin=74,vmax=79,aspect='auto');a.set_yticks(range(5),names);a.set_xticks(range(4),['A: 7d','B: +traffic','C: +meta','D: +words']);a.tick_params(length=0,labelsize=6,pad=3);a.spines[:].set_visible(False)
for i in range(5):
 for j in range(4):a.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=6.1)
a.set_xticks(np.arange(-.5,4,1),minor=True);a.set_yticks(np.arange(-.5,5,1),minor=True);a.grid(which='minor',color='white',lw=.6);a.tick_params(which='minor',length=0)
cax=fig.add_axes([.61,.895,.35,.023]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[74,76,79]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.4,length=1,pad=1);cb.ax.set_title('WAPE (%)',fontsize=5.8,pad=3)
b=fig.add_axes([.23,.15,.73,.27]);fig.text(.03,.49,'(b) Which update reduces error?',va='top',fontsize=7,fontweight='bold')
for term,color,marker,offset,label in [('numeric_origin','#6997ae','o',-.12,'Traffic/origin'),('words','#d29177','s',.12,'Words')]:
 z=np.array([d[k][term]['pp'] for k in keys]);lo=np.array([d[k][term]['CI95'][0] for k in keys]);hi=np.array([d[k][term]['CI95'][1] for k in keys]);b.errorbar(z,np.arange(5)+offset,xerr=[z-lo,hi-z],fmt=marker,color=color,ms=3,elinewidth=.8,capsize=2,label=label)
b.axvline(0,color='#aab2b7',ls='--',lw=.6);b.set_yticks(range(5),names);b.set_ylim(4.6,-.6);b.set_xlim(-.3,5.3);b.set_xticks([0,1,2,3,4,5]);b.set_xlabel('WAPE reduction (pp)',labelpad=3);b.tick_params(length=2,labelsize=6);b.tick_params(axis='y',length=0);b.legend(loc='upper center',bbox_to_anchor=(.52,-.30),frameon=False,ncol=2,fontsize=6,handlelength=1.8,columnspacing=1.3);fig.savefig(O/'information_updates.pdf');fig.savefig(O/'information_updates.png',dpi=350);print('information update figure complete')
