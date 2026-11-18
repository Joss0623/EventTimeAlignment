"""Exact grouped interventional Shapley, update integrated gradients, and cached layer traces.
Uses fixed validation-selected models; no training, parameter search, or outcome-based case selection.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import sys,json,time,math,csv,traceback,hashlib
from pathlib import Path
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from neural_readouts import net,load,read_npz,R,D,dump,snapshot_key
O=R/'interpretability01';O.mkdir(exist_ok=True)
KEYS=['Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B']
GROUPS={'Earlier counts (8-28d)':list(range(21)),'Recent counts (1-7d)':list(range(21,28)),'History scale':[28],'Reference forecast':list(range(29,36)),'Lead':[36],'Article metadata':list(range(37,40)),'Words':list(range(40,72))}
M=len(GROUPS);MASKS=np.array([[(s>>i)&1 for i in range(M)] for s in range(2**M)],dtype=bool)
FEATUREMASK=np.zeros((2**M,72),bool)
for i,indices in enumerate(GROUPS.values()):FEATUREMASK[:,indices]=MASKS[:,i,None]
OUTPUTS=['log1p_total_forecast_visits','mean_log_residual_correction']

def main():
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024)
 rows,x,y=load();data=read_npz(R/'neural_readouts/features.npz');tr=data['tr'];te=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']=='event' and z['lead']==1]);base=data['base'];saved=read_npz(R/'neural_readouts/all_predictions.npz');cfg=json.load(open(R/'neural_readouts/selected_configs.json'))
 assert len(te)==127
 old=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);retained=[str(read_npz(D/'representations'/(snapshot_key(z)+'.npz'))['text']) for z in rows];changed=np.array([retained[i]!=retained[old[i]] for i in te]);assert changed.sum()==43
 med=np.maximum(np.median(x,axis=1),1);rawf=np.c_[np.log1p(x)-np.log1p(med[:,None]),np.log1p(med),np.log1p(base)-np.log1p(med[:,None]),np.log1p([z['lead'] for z in rows])];sc=StandardScaler().fit(rawf[tr]);assert np.max(np.abs(sc.transform(rawf)-data['Metadata_0'][:,:37]))<1e-5
 mean=torch.tensor(sc.mean_,dtype=torch.float32);scale=torch.tensor(sc.scale_,dtype=torch.float32)
 bgpool=sorted([i for i in tr if rows[i]['kind']=='event' and rows[i]['lead']==1],key=lambda i:hashlib.sha256(('2024|shap|'+str(rows[i]['pageid'])).encode()).hexdigest());assert len({rows[i]['pageid'] for i in bgpool})==len(bgpool)
 backgrounds={'A':bgpool[:16],'B':bgpool[16:32]};caseorder=sorted([int(i) for i in te[changed]],key=lambda i:hashlib.sha256(('2024|case|'+str(rows[i]['pageid'])).encode()).hexdigest());cases=caseorder[:3];cases+= [int(i) for i in te if rows[i]['title']=='Kosmos 482' and i not in cases]
 protocol={'seed':2024,'training':False,'models':KEYS,'selection':'existing pooled-validation selected checkpoint per frozen family','explained_cases':'all127 test event lead1 windows; no error-based filtering','groups':GROUPS,'outputs':OUTPUTS,'background':'two disjoint sets of16 hash-selected training event pages at lead1; A primary, B sensitivity; same background identities for all models','method':'exact grouped interventional Shapley enumeration of128 coalitions, empirical mean over16 background pages; official SHAP ExactExplainer numerical cross-check','anchor':'reference forecast reconstructed from training-standardized numerical inputs; changes with numerical coalition so the anchored forecast is included','caveats':['interventional feature hybrids break cross-group correlations; these are model attributions, not causal effects','word features are PCA coordinates, not token-level explanations','prediction attribution is distinct from forecast accuracy gain'],'update_path':'64-point Gauss-Legendre integrated gradients from old7-day words to current words; current numerical inputs and metadata held fixed','case_selection':'first3 hash-selected changed test inputs, plus the previously reported Kosmos482 case; no new case selection by prediction error','layer_trace':'reuse two validation-selected LoRA traces on14 changed validation pages; layer norms/dimensions differ, sensitivity is not information sufficiency'}
 dump(O/'protocol.json',protocol);allout={};summary={};igsummary={};localcases={};crosschecks=[];weights=np.array([math.factorial(s.bit_count())*math.factorial(M-s.bit_count()-1)/math.factorial(M) if s.bit_count()<M else 0 for s in range(2**M)])
 for key in KEYS:
  v=cfg[key];features=data[key+'_'+str(v['dim'])];h=net(features.shape[1],v['width'],.1);path=R/'neural_readouts'/f"{key}_d{v['dim']}_w{v['width']}_lr{v['lr']:g}_{v['loss']}"/'best.pt';h.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));h.eval()
  def forward(z):
   correction=h(z);raw=z[:,:37]*scale+mean;logbase=raw[:,29:36]+raw[:,28:29];lp=torch.clamp(logbase+correction,0,25);total=torch.log1p(torch.expm1(lp.double()).sum(1));return torch.stack([total,correction.double().mean(1)],1)
  xt=torch.tensor(features[te]);
  with torch.inference_mode():out=forward(xt).numpy();cor=h(xt).numpy()
  p=np.maximum(np.expm1(np.clip(np.log1p(base[te])+cor,0,25)),0);assert np.max(np.abs(p-saved[key][te])/(1+saved[key][te]))<2e-5
  expected=np.log1p(saved[key][te].astype(np.float64).sum(1));assert np.max(np.abs(out[:,0]-expected))<5e-5
  stores={}
  for bank,bgids in backgrounds.items():
   bg=features[bgids];values=np.empty((len(te),2**M,2),np.float64)
   for j in range(len(te)):
    z=np.where(FEATUREMASK[:,None,:],features[te[j]][None,None,:],bg[None,:,:]).astype(np.float32).reshape(-1,72);pieces=[]
    with torch.inference_mode():
     for start in range(0,len(z),4096):pieces.append(forward(torch.tensor(z[start:start+4096])).numpy())
    values[j]=np.concatenate(pieces).reshape(2**M,len(bg),2).mean(1)
   phi=np.zeros((len(te),M,2),np.float64)
   for group in range(M):
    ss=[s for s in range(2**M) if not s&(1<<group)];phi[:,group]=np.einsum('s,nso->no',weights[ss],values[:,np.array(ss)+(1<<group)]-values[:,ss])
   residual=phi.sum(1)-(values[:,-1]-values[:,0]);assert np.max(np.abs(residual))<1e-8
   allout[key+'_'+bank+'_phi']=phi;allout[key+'_'+bank+'_coalitions']=values;stores[bank]=phi
   summary[key+'_'+bank]={'mean_absolute':np.abs(phi).mean(0).tolist(),'mean_signed':phi.mean(0).tolist(),'efficiency_max_error':float(np.max(np.abs(residual))),'background_rows':list(map(int,bgids))}
   print('EXACT SHAPLEY',key,bank,len(te),flush=True)
   if bank=='A':
    import shap
    for rowid in cases[:3]:
     j=int(np.flatnonzero(te==rowid)[0]);lookup=values[j]
     def game(present):return lookup[(np.rint(present).astype(int)*(2**np.arange(M))).sum(1)]
     explainer=shap.ExactExplainer(game,np.zeros((1,M)),feature_names=list(GROUPS));official=explainer(np.ones((1,M))).values[0];err=float(np.max(np.abs(official-phi[j])));assert err<1e-8;crosschecks.append({'model':key,'row':rowid,'max_abs_error':err,'shap_version':shap.__version__})
  summary[key+'_background_stability']={'rank_correlation_total':float(spearmanr(np.abs(stores['A']).mean(0)[:,0],np.abs(stores['B']).mean(0)[:,0]).statistic),'rank_correlation_correction':float(spearmanr(np.abs(stores['A']).mean(0)[:,1],np.abs(stores['B']).mean(0)[:,1]).statistic)}
  # Integrated gradients only along the actually observed word-feature update, without changing other feature groups.
  cur=features[te].copy();stale=cur.copy();stale[:,40:]=features[old[te],40:];delta=cur-stale;nodes,w=np.polynomial.legendre.leggauss(64);nodes=(nodes+1)/2;w=w/2;grads=np.zeros((len(te),72,2),np.float64)
  for node,weight in zip(nodes,w):
   z=torch.tensor(stale+float(node)*delta,requires_grad=True);vals=forward(z)
   for output in range(2):
    grad=torch.autograd.grad(vals[:,output].sum(),z,retain_graph=output==0)[0].detach().numpy();grads[:,:,output]+=weight*grad
  ig=grads*delta[:,:,None]
  with torch.inference_mode():current_output=forward(torch.tensor(cur)).numpy();old_output=forward(torch.tensor(stale)).numpy()
  change=current_output-old_output;complete=np.abs(ig.sum(1)-change);assert complete.max()<5e-5
  allout[key+'_update_ig']=ig;allout[key+'_update_output_change']=change
  l1=np.abs(ig[:,40:,:]).sum(1);cancellation=1-np.abs(ig[:,40:,:].sum(1))/np.maximum(l1,1e-15)
  igsummary[key]={'changed_n':int(changed.sum()),'mean_abs_update_total':float(np.abs(change[changed,0]).mean()),'median_abs_update_total':float(np.median(np.abs(change[changed,0]))),'mean_static_word_shap_total':float(np.abs(stores['A'][changed,6,0]).mean()),'mean_abs_update_correction':float(np.abs(change[changed,1]).mean()),'median_cancellation_total':float(np.median(cancellation[changed,0])),'max_completeness_error':float(complete.max()),'unchanged_max_output_change':float(np.abs(change[~changed]).max())}
  localcases[key]=[]
  for i in cases:
   j=int(np.flatnonzero(te==i)[0]);snap=json.load(open(D/'snapshots'/(snapshot_key(rows[i])+'.json')));old_snap=json.load(open(D/'snapshots'/(snapshot_key(rows[old[i]])+'.json')));localcases[key].append({'row':i,'pageid':rows[i]['pageid'],'title':rows[i]['title'],'origin':rows[i]['origin'],'current_revid':snap['revid'],'old_revid':old_snap['revid'],'group_shapley':stores['A'][j].tolist(),'expected_output':allout[key+'_A_coalitions'][j,0].tolist(),'current_output':current_output[j].tolist(),'old_word_output':old_output[j].tolist(),'word_ig_positive':np.maximum(ig[j,40:],0).sum(0).tolist(),'word_ig_negative':np.minimum(ig[j,40:],0).sum(0).tolist()})
  print('UPDATE IG',key,igsummary[key],flush=True)
 allout.update(test_rows=te,changed_mask=changed,group_names=np.array(list(GROUPS)),outputs=np.array(OUTPUTS));np.savez_compressed(O/'explanations.npz',**allout);dump(O/'group_shapley_summary.json',summary);dump(O/'update_ig_summary.json',igsummary);dump(O/'cases.json',localcases);dump(O/'official_shap_crosscheck.json',crosschecks)
 # Preserve sample-level traces; bootstrap paired validation pages without treating network layers as replicates.
 trace={}
 for family in ['Qwen3','Mistral7B']:
  src=R/'posttrain01/mechanism_ablation01'/f'{family}_lora_lr1e-05_layer_trace.json';d=json.load(open(src));pairs=[z for z in d['pair_traces']['adapter_on'] if z['split']=='val' and z['changed']];assert len(pairs)==14
  entries=[]
  for group in ['LLM_layers','head_layers']:
   for stage in pairs[0][group]:
    ar=np.array([z[group][stage]['relative_L2'] for z in pairs]);rng=np.random.default_rng(2024);means=np.array([ar[rng.integers(0,len(ar),len(ar))].mean() for _ in range(2000)]);entries.append({'stage':stage,'location':group,'n':len(ar),'mean_relative_L2':float(ar.mean()),'CI95':np.quantile(means,[.025,.975]).tolist()})
  trace[family]={'stages':entries,'knockouts':d['knockouts'],'scope':d['scope']}
 dump(O/'layer_sensitivity.json',trace)
 with open(O/'group_shapley.csv','w',newline='') as file:
  w=csv.writer(file);w.writerow(['model','background','row','pageid','group','log_total_forecast_phi','mean_log_correction_phi'])
  for key in KEYS:
   for bank in backgrounds:
    ar=allout[key+'_'+bank+'_phi']
    for j,i in enumerate(te):
     for group,name in enumerate(GROUPS):w.writerow([key,bank,int(i),rows[i]['pageid'],name,*ar[j,group]])
 dump(O/'complete.json',{'complete':True,'models':len(KEYS),'test_targets':len(te),'changed_targets':int(changed.sum()),'background_banks':2,'coalitions':2**M,'official_SHAP_checks':len(crosschecks),'seed':2024,'retrained':False});print('INTERPRETABILITY COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:dump(O/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
