"""Single-column results, five frozen LLM families; separate legend/colorbar regions."""
import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D
from run import load,error
R=Path('/home/home_new/qsmx/runs/AttentionPIT/pilot01');d=json.load(open(R/'results.json'));rows,x,y=load();p=np.load(R/'all_predictions.npz');leads=[7,3,1];ks=['Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B'];names=['Qwen3','Mistral','gpt-oss','DeepSeek','Kimi'];colors=['#6598B0','#D58F75','#9B8DB4','#83AC9D','#BEAA70'];markers=['o','s','^','D','v'];te=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']=='event']);lead=np.array([z['lead'] for z in rows]);idx=te[lead[te]==1]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],'font.size':6.2,'axes.labelsize':6.2,'axes.titlesize':7,'legend.fontsize':5.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(3.35,4.3));a=fig.add_axes([.16,.565,.33,.29]);b=fig.add_axes([.68,.565,.29,.29]);c=fig.add_axes([.16,.155,.33,.235]);e=fig.add_axes([.73,.155,.24,.235])
for xx,title in [(.16,'(a) Event error'),(.65,'(b) Text advantage')]:fig.text(xx,.976,title,fontsize=7,fontweight='bold',va='top')
for xx,title in [(.16,'(c) Event variation'),(.65,'(d) Text refresh')]:fig.text(xx,.46,title,fontsize=7,fontweight='bold',va='top')
cmap=LinearSegmentedColormap.from_list('error',['#80AFC1','#FAF7F1','#CD8877']);gainmap=LinearSegmentedColormap.from_list('gain',['#CD8877','#FAF7F1','#80AFC1']);keys=['Chronos2','NumericReadout','TFIDF']+ks;labs=['Chronos','Numeric','TF-IDF']+names;mat=np.array([[d['curves']['event'][str(l)][k]['WAPE'] for l in leads] for k in keys]);im=a.imshow(mat,cmap=cmap,vmin=65,vmax=82,aspect='auto');a.set_yticks(range(len(keys)),labs);a.set_xticks(range(3),['7','3','1']);a.set_xlabel('Lead (days)',labelpad=2);a.tick_params(length=0,pad=2)
for i in range(len(keys)):
 for j in range(3):a.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=5.4,color='#28313B')
a.set_xticks(np.arange(-.5,3,1),minor=True);a.set_yticks(np.arange(-.5,len(keys),1),minor=True);a.grid(which='minor',color='white',lw=.65);a.tick_params(which='minor',length=0)
ca=fig.add_axes([.19,.906,.27,.024]);cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[65,73,81]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.4,length=1,pad=1);cb.ax.set_title('WAPE (%)',fontsize=5.7,pad=3)
norm=TwoSlopeNorm(vmin=-3.2,vcenter=0,vmax=3.2)
for i,k in enumerate(ks):
 for j,l in enumerate(leads):
  ii=te[lead[te]==l];win=np.mean(error(y[ii],p[k][ii])<error(y[ii],p['NumericReadout'][ii]));g=d['gains']['event'][str(l)]['NumericReadout -> '+k]['pp'];b.scatter(j,i,s=100*win,c=[g],cmap=gainmap,norm=norm,edgecolor='#46505A',linewidth=.4);b.text(j,i+.36,f'{g:+.2f}',ha='center',va='center',fontsize=5)
b.set(xlim=(-.5,2.5),ylim=(4.65,-.65));b.set_xticks(range(3),['7','3','1']);b.set_yticks(range(5),names);b.set_xlabel('Lead (days)',labelpad=2);b.tick_params(length=0,pad=2)
ca=fig.add_axes([.69,.906,.27,.024]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=gainmap),cax=ca,orientation='horizontal',ticks=[-3,0,3]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.4,length=1,pad=1);cb.ax.set_title('Gain (pp)',fontsize=5.7,pad=3)
# Independent legend strip: circles and labels never share the title's baseline.
la=fig.add_axes([.65,.477,.33,.044]);la.set_axis_off();la.set(xlim=(0,1),ylim=(0,1));la.text(.5,.98,'Pages improved',ha='center',va='top',fontsize=5.4)
for xx,f in zip([.08,.42,.77],[.25,.5,.75]):la.scatter(xx,.15,s=100*f,facecolor='white',edgecolor='#64727B',lw=.4,clip_on=False);la.text(xx+.09,.15,f'{f:.0%}',ha='left',va='center',fontsize=5.2)
en=error(y[idx],p['NumericReadout'][idx]);upper=max(110,max(float(error(y[idx],p[k][idx]).max()) for k in ks+['NumericReadout'])*1.05)
for k,n,col,mark in zip(ks,names,colors,markers):c.scatter(en,error(y[idx],p[k][idx]),s=11,color=col,marker=mark,edgecolor='#4D5960',lw=.25,alpha=.8)
c.plot([0,upper],[0,upper],ls='--',color='#A3ADB3',lw=.6,zorder=0);c.set(xlim=(0,upper),ylim=(0,upper),xlabel='Numeric WAPE (%)',ylabel='LLM WAPE (%)');c.set_xticks([0,50,100]);c.set_yticks([0,50,100])
for j,(k,n,col,mark) in enumerate(zip(ks,names,colors,markers)):
 z=d['gains']['event']['1'][k+'Frozen7d -> '+k];mean=z['pp'];lo,hi=z['CI95'];e.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt=mark,color=col,ecolor=col,ms=3,capsize=2,elinewidth=.8,markeredgecolor='#4D5960',markeredgewidth=.3)
e.axvline(0,color='#9CA6AC',ls='--',lw=.6);e.set_yticks(range(5),names);e.set_ylim(4.6,-.6);e.set_xlim(-.45,.45);e.set_xticks([-.4,0,.4]);e.set_xlabel('WAPE gain (pp)',labelpad=2);e.tick_params(axis='y',length=0,pad=2);e.grid(axis='x',alpha=.12,lw=.4)
# Shared categorical legend is below all axes, outside labels and data.
handles=[Line2D([],[],ls='',marker=m,color=co,markersize=3,label=n,markeredgecolor='#4D5960',markeredgewidth=.3) for n,co,m in zip(names,colors,markers)];fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.003),ncol=3,frameon=False,columnspacing=1,handletextpad=.25,labelspacing=.5)
for ax in [a,b,c,e]:ax.tick_params(labelsize=5.8);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.savefig(R/'profiles.pdf');fig.savefig(R/'profiles.png',dpi=450);print('COMPLETE: 5 LLM families, 4 panels, separate colorbar and legend strips')
