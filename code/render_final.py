"""Single-column final result figures from completed neural experiment only."""
import json,csv,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from run import load,error,R
O=R/'neural_readouts';d=json.load(open(O/'results.json'));p=np.load(O/'all_predictions.npz');rows,x,y=load();ks=['Chronos2','PatchTST','iTransformer','TimeMixer','Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B'];labs=['Chronos-2','PatchTST','iTransformer','TimeMixer','Qwen3','Mistral','gpt-oss','DeepSeek','Kimi'];ll=ks[4:];names=labs[4:];leads=[7,3,1];cols=['#6598B0','#D58F75','#9B8DB4','#83AC9D','#BEAA70'];marks=['o','s','^','D','v'];ii=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']=='event']);lead=np.array([z['lead'] for z in rows]);errmap=LinearSegmentedColormap.from_list('error',['#80AFC1','#FAF7F1','#CD8877']);gainmap=LinearSegmentedColormap.from_list('gain',['#CD8877','#FAF7F1','#80AFC1'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],'font.size':6.2,'axes.labelsize':6.2,'axes.titlesize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(3.35,2.6));a=fig.add_axes([.20,.25,.30,.54]);b=fig.add_axes([.72,.25,.26,.54]);fig.text(.17,.985,'(a) Event errors',va='top',fontsize=7,fontweight='bold');fig.text(.66,.985,'(b) Text gains',va='top',fontsize=7,fontweight='bold');mat=np.array([[d['curves']['event'][str(l)][k]['WAPE'] for l in leads] for k in ks]);im=a.imshow(mat,cmap=errmap,vmin=np.floor(mat.min()/5)*5,vmax=np.ceil(mat.max()/5)*5,aspect='auto');a.set_yticks(range(9),labs);a.set_xticks(range(3),['7','3','1']);a.tick_params(length=0,pad=2,labelsize=6.1);a.set_xlabel('Lead (days)',labelpad=2)
for i in range(9):
 for j in range(3):a.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=6.0,color='#28313B')
a.set_xticks(np.arange(-.5,3,1),minor=True);a.set_yticks(np.arange(-.5,9,1),minor=True);a.grid(which='minor',color='white',lw=.6);a.tick_params(which='minor',length=0);ca=fig.add_axes([.20,.865,.30,.028]);cb=fig.colorbar(im,cax=ca,orientation='horizontal');cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.2,length=1,pad=1);cb.ax.set_title('WAPE (%)',fontsize=5.7,pad=2)
gains=np.array([[d['gains']['event'][str(l)]['NumericReadout -> '+k]['pp'] for l in leads] for k in ll]);lim=max(1,float(np.ceil(np.abs(gains).max())));norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
for i,k in enumerate(ll):
 for j,l in enumerate(leads):
  ix=ii[lead[ii]==l];fraction=np.mean(error(y[ix],p[k][ix])<error(y[ix],p['NumericReadout'][ix]));g=gains[i,j];b.scatter(j,i,s=100*fraction,c=[g],cmap=gainmap,norm=norm,edgecolor='#46505A',linewidth=.4);b.text(j,i+.46,f'{g:+.2f}',ha='center',va='center',fontsize=5.7)
b.set(xlim=(-.5,2.5),ylim=(4.65,-.6));b.set_xticks(range(3),['7','3','1']);b.set_yticks(range(5),names);b.tick_params(length=0,pad=2,labelsize=6.1);b.set_xlabel('Lead (days)',labelpad=2);ca=fig.add_axes([.71,.865,.27,.028]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=gainmap),cax=ca,orientation='horizontal',ticks=[-lim,0,lim]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.2,length=1,pad=1);cb.ax.set_title('Gain (pp)',fontsize=5.7,pad=2)
la=fig.add_axes([.49,.015,.49,.105]);la.set_axis_off();la.set(xlim=(0,1),ylim=(0,1));la.text(.5,.98,'Pages improved',ha='center',va='top',fontsize=5.4)
for xx,f in zip([.08,.4,.73],[.25,.5,.75]):la.scatter(xx,.25,s=100*f,facecolor='white',edgecolor='#64727B',lw=.4,clip_on=False);la.text(xx+.08,.25,f'{f:.0%}',ha='left',va='center',fontsize=5.2)
fig.savefig(O/'event_profiles.pdf');fig.savefig(O/'event_profiles.png',dpi=450)
fig,ax=plt.subplots(1,2,figsize=(3.35,1.9));fig.subplots_adjust(left=.19,right=.99,bottom=.25,top=.8,wspace=.65)
for panel,suffix,title in zip(ax,['FixedMismatch','Frozen7d'],['(a) Text identity','(b) Text refresh']):
 bounds=[]
 for j,(k,col,mark) in enumerate(zip(ll,cols,marks)):
  z=d['gains']['event']['1'][k+suffix+' -> '+k];m=z['pp'];lo,hi=z['CI95'];bounds.extend([lo,hi]);panel.errorbar(m,j,xerr=[[m-lo],[hi-m]],fmt=mark,color=col,ms=3,elinewidth=.8,capsize=2,markeredgecolor='#4D5960',markeredgewidth=.3)
 span=max(max(bounds)-min(bounds),.05);panel.set_xlim(min(0,min(bounds))-.1*span,max(0,max(bounds))+.1*span);panel.axvline(0,ls='--',color='#A3ADB3',lw=.6);panel.set_yticks(range(5),names);panel.set_ylim(4.6,-.6);panel.set_xlabel('WAPE gain (pp)',labelpad=2);panel.set_title(title,loc='left',fontsize=6.7,pad=7,fontweight='bold');panel.tick_params(labelsize=5.6,length=2);panel.tick_params(axis='y',length=0);panel.grid(axis='x',alpha=.1,lw=.4)
fig.savefig(O/'text_controls.pdf');fig.savefig(O/'text_controls.png',dpi=450)
with (O/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','event_WAPE_7d','event_WAPE_3d','event_WAPE_1d','calendar_WAPE_7d','calendar_WAPE_3d','calendar_WAPE_1d','fresh_event_WAPE_1d','event_RMSLE_1d'])
 for k,n in zip(ks,labs):w.writerow([n]+[d['curves'][kind][str(l)][k]['WAPE'] for kind in ['event','calendar_control'] for l in leads]+[d['curves']['fresh_event']['1'][k]['WAPE'],d['curves']['event']['1'][k]['RMSLE']])
print('SINGLE-COLUMN RESULT FIGURES AND COMPARISON CSV COMPLETE')
