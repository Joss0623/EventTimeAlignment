"""Decision-cost correction: fit expected regret of fixed forecast actions, not an event-label gate.
All context is as-of; no event/calendar kind or outcomes enter inference features.
"""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import sys,json,time,csv,argparse,traceback,subprocess,hashlib
from pathlib import Path
import numpy as np,torch
from posttrain_lora import design,O,R,D,C,error,read_npz,dump,snapshot_key
from mechanism_ablation import head_for,delta_controls
from run import load
from sklearn.tree import DecisionTreeRegressor,export_text
Q=O/'conditional_correction01';Q.mkdir(exist_ok=True)
ACTIONS=np.array([.5,.75,1.,1.25,1.5,2.,3.],np.float32);MARGINS=[0.,.005,.01,.02,.05]
LRS=[1e-5,3e-5,1e-4]
CONDITIONS=['risk_Numeric','risk_Metadata']+[f'risk_{s}_{f}' for f in ['Qwen3','Mistral7B'] for s in ['Current','Delta','Old','Wrong']]+[f'{loss}_{s}' for loss in ['wape','huber'] for s in ['Numeric','Metadata','Qwen3','Mistral7B']]

def context():
 rows,f,base,y,target,texts,tr,va=design();_,x,_=load();old=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);snap=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];donors=[]
 for i,z in enumerate(rows):
  pool=[j for j in tr if rows[j]['pageid']!=z['pageid'] and snap[j]['available'] and snap[j]['revision_time']<z['origin'] and rows[j]['origin']<=z['origin']]
  donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else -1)
 return rows,f,base,y,target,texts,tr,va,x,old,snap,donors

def diagnostic():
 rows,f,b,y,t,texts,tr,va,x,old,snap,donors=context();p=read_npz(O/'mechanism_ablation01/predictions.npz')['Mistral7B_CurrentText128'];gain=error(y,b)-error(y,p)
 a=np.c_[np.log((x[:,-3:].mean(1)+1)/(np.median(x[:,:-7],1)+1)),np.log((x[:,-3:].mean(1)+1)/(x[:,-7:-3].mean(1)+1)),np.log((b.mean(1)+1)/(x[:,-7:].mean(1)+1)),np.std(np.diff(np.log1p(x),axis=1),axis=1),np.log1p(b.mean(1)),np.log1p([r['lead'] for r in rows]),f[:,37:]];names=['recent_level','recent_growth','forecast_recent','volatility','logbase','loglead','textlen','age','available'];tree=DecisionTreeRegressor(max_depth=3,min_samples_leaf=60,random_state=2024).fit(a[tr],gain[tr]);scores=tree.predict(a);ii=va[scores[va]>0]
 result={'source':'completed128Mistral correction; train-only depth3 minleaf60 diagnostic; depth1/2 also inspected, no claim of independent model selection','tree':export_text(tree,feature_names=names),'train_rows':len(tr),'validation_rows':len(va),'validation_selected_n':len(ii),'validation_conditional_gain_pp':float(gain[ii].mean()) if len(ii) else None,'validation_total_gain_pp':float(np.mean(np.where(scores[va]>0,gain[va],0))),'features_no_future_or_kind':names,'train_val_page_disjoint':not bool(set(rows[i]['pageid'] for i in tr)&set(rows[i]['pageid'] for i in va)),'test_outcomes_used':False}
 dump(Q/'asof_diagnostic.json',result)

def inputs(condition,c,raws,variant='Current'):
 rows,f,b,y,t,texts,tr,va,x,old,snap,donors=c;f=f.copy();zero=np.zeros_like(raws['Qwen3']);words=zero
 if condition.endswith('Numeric'):f[:,37:]=0
 family=next((k for k in ['Qwen3','Mistral7B'] if k in condition),None)
 if family:
  raw=raws[family]
  mode=condition.split('_')[1] if condition.startswith('risk_') else 'Current'
  if variant!='Current':mode=variant
  if mode=='Current':words=raw
  elif mode=='Old':words=raw[old]
  elif mode=='Wrong':words=np.array([raw[j] if j>=0 else np.zeros(raw.shape[1],np.float32) for j in donors])
  elif mode=='Delta':words=raw-raw[old]
  elif mode=='Zero':words=zero
  else:raise ValueError(mode)
 return f.astype(np.float32),words.astype(np.float32)

def risk_predictions(values,base,margin):
 cost=values.copy();cost[:,2]=0.;choice=np.argmin(cost,axis=1);choice[np.min(cost,axis=1)>=-margin]=2
 return base*ACTIONS[choice,None],choice

def evaluate(h,fx,emb,b,y,ids,risk):
 h.eval()
 with torch.inference_mode():v=h['predict'](torch.cat([fx[ids],h['project'](emb[ids])],-1)).cpu().numpy()
 if risk:
  trials=[(float(error(y[ids],risk_predictions(v,b[ids],m)[0]).mean()),m) for m in MARGINS];score,margin=min(trials,key=lambda q:(q[0],-q[1]));return score,margin,v
 p=np.maximum(np.expm1(np.clip(np.log1p(b[ids])+v,0,25)),0);return float(error(y[ids],p).mean()),None,v

def worker(jobsfile):
 torch.set_num_threads(2);c=context();rows,f,b,y,t,texts,tr,va,x,old,snap,donors=c;raws={k:read_npz(O/(k+'_nf4_frozen_embeddings.npz'))['embedding'] for k in ['Qwen3','Mistral7B']};device='cuda';yy=torch.tensor(y,device=device);bb=torch.tensor(np.log1p(b),device=device);tt=torch.tensor(t,device=device)
 # Costs are training targets only. Future test/validation values never enter training updates.
 costs=np.stack([error(y[tr],b[tr]*a)/100 for a in ACTIONS],1);costs-=costs[:,2:3];ct=torch.zeros((len(rows),7),device=device);ct[tr]=torch.tensor(costs,device=device)
 for cfg in json.load(open(jobsfile)):
  condition,lr=cfg['condition'],cfg['lr'];dest=Q/f'{condition}_lr{lr:g}';dest.mkdir(exist_ok=True)
  if (dest/'result.json').exists():continue
  torch.manual_seed(2024);np.random.seed(2024);h=head_for(4096,32).cuda();torch.nn.init.zeros_(h['predict'][-1].weight);torch.nn.init.zeros_(h['predict'][-1].bias);ff,ee=inputs(condition,c,raws);fx=torch.tensor(ff,device=device);emb=torch.tensor(ee,device=device);risk=condition.startswith('risk_');opt=torch.optim.AdamW(h.parameters(),lr=lr,weight_decay=1e-4);rng=np.random.default_rng(2024);best=np.inf;stale=0;curve=[];bestepoch=0;bestmargin=None
  # Every correction path must reproduce the baseline before any training.
  initial,initialmargin,v=evaluate(h,fx,emb,b,y,va,risk);assert abs(initial-float(error(y[va],b[va]).mean()))<.001
  dump(dest/'started.json',{'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'seed':2024,'condition':condition,'lr':lr,'initial_validation_WAPE':initial,'parameters':sum(p.numel() for p in h.parameters()),'min_epochs':30,'max_epochs':150,'patience':20,'batch':16,'microbatch':16,'test_used_in_training_or_selection':False})
  for epoch in range(1,151):
   h.train();losses=[]
   perm=rng.permutation(tr)
   for start in range(0,len(perm),16):
    ii=perm[start:start+16];opt.zero_grad(set_to_none=True);v=h['predict'](torch.cat([fx[ii],h['project'](emb[ii])],-1))
    if risk:loss=((v[:,[0,1,3,4,5,6]]-ct[ii][:,[0,1,3,4,5,6]])**2).mean()
    elif condition.startswith('huber_'):loss=torch.nn.functional.smooth_l1_loss(v,tt[ii])
    else:
     p=torch.expm1(torch.clamp(bb[ii]+v,0,25));loss=(torch.abs(p-yy[ii]).sum(1)/torch.clamp(yy[ii].sum(1),min=1.)).mean()
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step();losses.append(float(loss.detach()))
   score,margin,_=evaluate(h,fx,emb,b,y,va,risk);curve.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_WAPE':score,'validation_margin':margin})
   if score<best-1e-8:best=score;bestepoch=epoch;bestmargin=margin;stale=0;torch.save(h.state_dict(),dest/'head.pt')
   else:stale+=1
   dump(dest/'curve.json',curve)
   if epoch>=30 and stale>=20:break
  dump(dest/'result.json',{'condition':condition,'lr':lr,'seed':2024,'epochs_trained':epoch,'best_epoch':bestepoch,'validation_WAPE':best,'margin':bestmargin,'checkpoint':str(dest/'head.pt'),'loss':'MSE of per-action incremental WAPE risk' if risk else 'macroWAPE' if condition.startswith('wape_') else 'logHuber','test_evaluated':False});print('COMPLETE',condition,lr,epoch,best,flush=True)
 print('BATCH COMPLETE',jobsfile,flush=True)

def report():
 torch.set_num_threads(2);c=context();rows,f,b,y,t,texts,tr,va,x,old,snap,donors=c;raws={k:read_npz(O/(k+'_nf4_frozen_embeddings.npz'))['embedding'] for k in ['Qwen3','Mistral7B']};selected={}
 for condition in CONDITIONS:
  files=list(Q.glob(condition+'_lr*/result.json'));assert len(files)==3,(condition,len(files));selected[condition]=min((json.load(open(p)) for p in files),key=lambda v:v['validation_WAPE'])
 # Commit all model choices before evaluating any test label in this experiment.
 global_scale=min(ACTIONS,key=lambda a:float(error(y[va],b[va]*a).mean()));strong=read_npz(R/'neural_readouts/all_predictions.npz');baseline={k:strong[k] for k in ['Chronos2','PatchTST','iTransformer','TimeMixer','NumericReadout']};baseline['GlobalValidationScale']=b*global_scale
 selection={'seed':2024,'selected':selected,'global_scale':float(global_scale),'selection':'pooled validation macroWAPE; no test-directed changes','test_prior_inspection':'same test used in previous project analyses; this is exploratory, not a pristine new holdout'};dump(Q/'selection_locked.json',selection)
 pred=dict(baseline);actions={};counter={}
 for condition,v in selected.items():
  h=head_for(4096,32);h.load_state_dict(torch.load(v['checkpoint'],map_location='cpu',weights_only=True));h.eval();ff,ee=inputs(condition,c,raws)
  with torch.inference_mode():delta=h['predict'](torch.cat([torch.tensor(ff),h['project'](torch.tensor(ee))],-1)).numpy()
  if condition.startswith('risk_'):pred[condition],actions[condition]=risk_predictions(delta,b,v['margin'])
  else:pred[condition]=np.maximum(np.expm1(np.clip(np.log1p(b)+delta,0,25)),0)
  assert abs(float(error(y[va],pred[condition][va]).mean())-v['validation_WAPE'])<.005,(condition,v)
  if any(k in condition for k in ['Qwen3','Mistral7B']) and '_Old_' not in condition and '_Wrong_' not in condition:
   for variant in ['Old','Wrong']:
    ff,ee=inputs(condition,c,raws,variant)
    # Delta model fixed-old means zero innovation; wrong control is norm-matched wrong innovation.
    if '_Delta_' in condition:
     family=condition.split('_')[-1]
     if variant=='Old':ee=np.zeros_like(ee)
     else:ee=delta_controls(raws[family],old,rows,tr,snap,texts)[1]
    with torch.inference_mode():out=h['predict'](torch.cat([torch.tensor(ff),h['project'](torch.tensor(ee))],-1)).numpy()
    label=condition+'_Fixed'+variant
    if condition.startswith('risk_'):pred[label],actions[label]=risk_predictions(out,b,v['margin'])
    else:pred[label]=np.maximum(np.expm1(np.clip(np.log1p(b)+out,0,25)),0)
 # Fixed validation-selected numerical champion includes new equally trained controls.
 numeric_names=list(baseline)+['risk_Numeric','risk_Metadata','wape_Numeric','wape_Metadata','huber_Numeric','huber_Metadata'];numeric_champion=min(numeric_names,key=lambda k:float(error(y[va],pred[k][va]).mean()));risk_names=[k for k in selected if k.startswith('risk_')];risk_champion=min(risk_names,key=lambda k:selected[k]['validation_WAPE']);text_names=[k for k in risk_names if any(v in k for v in ['Qwen3','Mistral7B']) and ('_Current_' in k or '_Delta_' in k)];text_champion=min(text_names,key=lambda k:selected[k]['validation_WAPE']);selection.update(numeric_champion=numeric_champion,risk_champion=risk_champion,text_risk_champion=text_champion);dump(Q/'selection_locked.json',selection)
 subsets={'test_all':np.array([i for i,z in enumerate(rows) if z['split']=='test']),'test_event1':np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']=='event' and z['lead']==1]),'test_calendar1':np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['kind']=='calendar_control' and z['lead']==1])};fresh=set(json.load(open(R/'protocol.json'))['fresh_test_pageids']);subsets['test_fresh_event1']=np.array([i for i in subsets['test_event1'] if rows[i]['pageid'] in fresh]);table=[];comparisons={}
 for name,p in pred.items():
  row={'model':name,'validation_WAPE':float(error(y[va],p[va]).mean())}
  for s,ii in subsets.items():row[s+'_WAPE']=float(error(y[ii],p[ii]).mean())
  if name in selected:row.update(lr=selected[name]['lr'],epochs=selected[name]['epochs_trained'],best_epoch=selected[name]['best_epoch'])
  table.append(row)
 for s,ii in subsets.items():
  groups=np.array([rows[i]['pageid'] for i in ii]);ug=np.unique(groups);rng=np.random.default_rng(2024);boots=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(ug,len(ug),replace=True)]) for _ in range(2000)];es={k:error(y[ii],p[ii]) for k,p in pred.items()};pairs=[(numeric_champion,k) for k in selected]+[(k+'_Fixed'+a,k) for k in selected if k+'_FixedOld' in es for a in ['Old','Wrong']]+[('NumericReadout',risk_champion),('NumericReadout',text_champion),('risk_Numeric',text_champion)];comparisons[s]={}
  for first,second in pairs:
   if first not in es:continue
   d=es[first]-es[second];comparisons[s][first+' -> '+second]={'gain_pp':float(d.mean()),'CI95':np.quantile([d[j].mean() for j in boots],[.025,.975]).tolist(),'n':len(ii),'pages':len(ug)}
 # Counts/actions are descriptive only; event vs control status never entered features/training gate.
 action_summary={k:{s:{str(float(a)):int((v[ii]==j).sum()) for j,a in enumerate(ACTIONS)} for s,ii in subsets.items()} for k,v in actions.items() if k in selected}
 results={'seed':2024,'selection':selection,'comparison':table,'paired_gains':comparisons,'action_counts':action_summary,'sample_counts':{k:len(v) for k,v in subsets.items()},'interpretation':'test conclusions must compare against validation-selected numerical champion and fixed-old/wrong counterfactual inputs; training MSE here estimates decision costs, not forecast-value error','limitations':['single seed','same historical test previously inspected','retrospective event/control cohort, not deployment warning prevalence','no guarantee novel contribution from a generic cost predictor']};dump(Q/'results.json',results);np.savez_compressed(Q/'predictions.npz',**pred)
 with open(Q/'comparison.csv','w',newline='') as file:
  columns=['model','validation_WAPE','test_all_WAPE','test_event1_WAPE','test_calendar1_WAPE','test_fresh_event1_WAPE','lr','epochs','best_epoch'];w=csv.DictWriter(file,fieldnames=columns);w.writeheader();w.writerows(table)
 dump(Q/'complete.json',{'complete':True,'configs':len(CONDITIONS)*len(LRS),'minimum_epochs':min(json.load(open(p))['epochs_trained'] for p in Q.glob('*_lr*/result.json'))});dump(Q/'state.json',{'stage':'complete','completed':len(CONDITIONS)*len(LRS),'total':len(CONDITIONS)*len(LRS)});print('FULL EXPERIMENT COMPLETE',numeric_champion,risk_champion,text_champion,flush=True)

def manage():
 protocol={'seed':2024,'question':'Can as-of context predict when corrections improve rather than harm the same numerical forecast?','model':'single matched MLP estimates per-action excess forecast risk; retain unchanged baseline action; no stacked gate or auxiliary event classifier','actions':ACTIONS.tolist(),'margins_validation_only':MARGINS,'conditions':CONDITIONS,'lrs':LRS,'configs':len(CONDITIONS)*len(LRS),'budget':{'min_epochs':30,'max_epochs':150,'patience':20,'batch':16,'microbatch':16,'weight_decay':1e-4,'gradient_clip':1},'inputs':'same original retained text and cached frozen NF4 Qwen/Mistral; no model download or new article information','controls':'numeric/metadata cost models; same capacity direct WAPE and logHuber correction; retrained stale and wrong text; fixed-weight stale/mismatch and norm-matched wrong innovation','selection':'all checkpoints/LR/margins by validation pooled WAPE; report every condition on full test after selection locked','inference_exclusions':['future event/control kind','future observations','page identity','event_id'],'test_status':'prior test inspected in project; no new pristine holdout claim','parameter_count':148583,'initialization':'all final layers zero; exact baseline identity for every condition','diagnostic_source':'train/validation only shallow context analysis; no future-label gate at inference'}
 dump(Q/'protocol.json',protocol);diagnostic();jobs=[{'condition':k,'lr':lr} for k in CONDITIONS for lr in LRS if not (Q/f'{k}_lr{lr:g}'/'result.json').exists()];gpu_info=[list(map(int,z.split(','))) for z in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()];gpus=[g for g,free in gpu_info if g in [0,1,2,3] and free>=3000];assert gpus,'No free research head GPU; do not interrupt other users'
 active=[]
 for index,gpu in enumerate(gpus):
  q=jobs[index::len(gpus)]
  if not q:continue
  path=Q/f'queue_gpu{gpu}.json';dump(path,q);log=open(Q/f'gpu{gpu}.log','a');p=subprocess.Popen([sys.executable,'-u',str(C/'conditional_correction.py'),'worker','--jobs',str(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();active.append((p,gpu))
 while active:
  for p,gpu in list(active):
   if p.poll() is not None:
    if p.returncode:raise RuntimeError(f'worker GPU{gpu} failed; inspect log before resuming')
    active.remove((p,gpu))
  dump(Q/'state.json',{'stage':'full_training','completed':len(list(Q.glob('*_lr*/result.json'))),'total':len(CONDITIONS)*len(LRS),'active':{str(g):p.pid for p,g in active}})
  if active:time.sleep(10)
 report()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['manage','worker','report']);p.add_argument('--jobs');args=p.parse_args()
 try:
  if args.stage=='manage':manage()
  elif args.stage=='worker':worker(args.jobs)
  else:report()
 except Exception as e:dump(Q/(args.stage+('_'+Path(args.jobs).stem if args.jobs else '')+'_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
