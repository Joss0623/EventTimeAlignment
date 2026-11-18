import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
import sys,json,time,hashlib,itertools,subprocess,traceback,argparse
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';D=B/'data/AttentionPIT/alert01';R=B/'runs/AttentionPIT/alert01';PY=B/'venvs/attentionpit/bin/python'
def dump(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def data():
 a=dict(np.load(D/'panel.npz'));rows=json.load(open(D/'rows.json'));a['split']=np.array([r['split'] for r in rows]);return rows,a
class Model(torch.nn.Module):
 def __init__(self,key,w,nf):
  super().__init__();self.key=key
  if key in ['PatchTST','iTransformer','TimeMixer']:
   from deep_baselines import config
   sys.path.insert(0,str(B/'pycodes/Time-Series-Library'));cfg=config(key,w);cfg.task_name='classification';cfg.num_class=1
   mod=__import__('models.'+key,fromlist=['Model']);self.net=mod.Model(cfg,patch_len=7,stride=7) if key=='PatchTST' else mod.Model(cfg);self.scale=torch.nn.Linear(nf,1)
  else:self.net=torch.nn.Sequential(torch.nn.Linear(nf,w),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(w,w),torch.nn.GELU(),torch.nn.Linear(w,1))
 def forward(self,x,f):
  if self.key in ['PatchTST','iTransformer','TimeMixer']:return self.net(x[:,:,None],torch.ones_like(x),None,None).ravel()+self.scale(f).ravel()
  return self.net(f).ravel()
def features(key,rows,a):
 tr=np.flatnonzero(a['split']=='train');x=a['x'];lx=np.log1p(x);med=np.maximum(np.median(x,axis=1),1);dow=np.array([r['ordinal']%7 for r in rows]);yr=np.array([r['ordinal']%365.25 for r in rows]);base=np.c_[lx-np.log1p(med[:,None]),np.log1p(med),np.log1p(x[:,-1])-np.log1p(med),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),np.sin(2*np.pi*yr/365.25),np.cos(2*np.pi*yr/365.25)]
 meta=StandardScaler().fit(a['meta'][tr]).transform(a['meta']);q=np.zeros((len(rows),32),dtype=np.float32)
 if key not in ['Numeric','Metadata','PatchTST','iTransformer','TimeMixer']:
  q=np.load(D/(key+'_features.npy'))
 extra=np.c_[np.zeros_like(meta) if key in ['Numeric','PatchTST','iTransformer','TimeMixer','Chronos2'] else meta,q]
 f=StandardScaler().fit(base[tr]).transform(base);return np.c_[f,extra].astype(np.float32),((lx-lx[tr].mean())/max(lx[tr].std(),1e-6)).astype(np.float32)
def work(key,w,lr,pw):
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 dest=R/'trials'/f'{key}_w{w}_lr{lr:g}_pw{pw}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return
 rows,a=data();f,x=features(key,rows,a);tr=np.flatnonzero(a['split']=='train');va=np.flatnonzero(a['split']=='val');xt=torch.tensor(x,device='cuda');ft=torch.tensor(f,device='cuda');yt=torch.tensor(a['y'].astype(np.float32),device='cuda');model=Model(key,w,f.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);weight=1 if pw==0 else float(np.sqrt((len(tr)-a['y'][tr].sum())/max(1,a['y'][tr].sum())));gen=np.random.default_rng(2024);best=-1;stale=0;curve=[]
 for epoch in range(1,151):
  model.train();total=0;seen=0
  for ids in np.array_split(gen.permutation(tr),max(1,int(np.ceil(len(tr)/512)))):
   opt.zero_grad(set_to_none=True);z=model(xt[ids],ft[ids]);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yt[ids],pos_weight=torch.tensor(weight,device='cuda'));assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
  model.eval()
  with torch.inference_mode():p=np.concatenate([model(xt[ii],ft[ii]).sigmoid().cpu().numpy() for ii in np.array_split(va,max(1,int(np.ceil(len(va)/1024))))])
  score=float(average_precision_score(a['y'][va],p));curve.append({'epoch':epoch,'loss':total/seen,'validation_AP':score})
  if score>best+1e-8:best=score;be=epoch;stale=0;torch.save(model.state_dict(),dest/'best.pt')
  else:stale+=1
  dump(dest/'curve.json',curve)
  if epoch%5==0:print(dest.name,epoch,score,flush=True)
  if epoch>=30 and stale>=20:break
 model.load_state_dict(torch.load(dest/'best.pt',map_location='cuda',weights_only=True));model.eval()
 with torch.inference_mode():p=np.concatenate([model(xt[ii],ft[ii]).sigmoid().cpu().numpy() for ii in np.array_split(np.arange(len(rows)),max(1,int(np.ceil(len(rows)/1024))))])
 assert np.isfinite(p).all();np.save(dest/'predictions.npy',p);dump(dest/'result.json',{'key':key,'width':w,'lr':lr,'positive_weight_mode':pw,'positive_weight':weight,'epochs_trained':epoch,'best_epoch':be,'validation_AP':best,'seed':2024});print('DONE',dest.name,flush=True)
def threshold(y,p,budget=.01):
 neg=np.sort(p[y==0]);allowed=int(np.floor(budget*len(neg)));return float(np.nextafter(neg[-allowed-1],np.inf)) if len(neg) else 1.01
def metric(rows,a,p,idx,thr,quiet=False):
 ii=idx[a['quiet'][idx]] if quiet else idx;y=a['y'][ii];alarm=p[ii]>=thr;events={(rows[i]['page'],int(a['first'][i])):[] for i in idx if a['first'][i]>=0}
 for i,hit in zip(ii,alarm):
  if a['first'][i]<0:continue
  k=(rows[i]['page'],int(a['first'][i]));events.setdefault(k,[])
  if hit:events[k].append(int(a['first'][i]-rows[i]['ordinal']))
 detected=np.array([bool(v) for v in events.values()]);leads=np.array([max(v) if v else 0 for v in events.values()]);return {'n_origins':len(ii),'positives':int(y.sum()),'AP':float(average_precision_score(y,p[ii])) if y.sum() else None,'FPR_percent':float(100*alarm[y==0].mean()) if (y==0).any() else None,'precision_percent':float(100*y[alarm].mean()) if alarm.any() else 0,'event_count':len(events),'event_recall_percent':float(100*detected.mean()) if len(leads) else None,'mean_lead_all_events_days':float(leads.mean()) if len(leads) else None,'mean_lead_detected_days':float(leads[detected].mean()) if detected.any() else None,'threshold':thr}
def report():
 rows,a=data();va=np.flatnonzero(a['split']=='val');te=np.flatnonzero(a['split']=='test');out={'seed':2024,'data':json.load(open(R/'data_summary.json')),'models':{},'selection':'validation AP only; validation negative-day FPR1% threshold','budgets':{}};pred={}
 keys=['PatchTST','iTransformer','TimeMixer','Numeric','Metadata','TFIDF','Qwen3','Mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct','Chronos2']
 for key in keys:
  paths=list((R/'trials').glob(key+'_*/result.json'))
  if len(paths)!=12:continue
  selected=max(paths,key=lambda p:json.load(open(p))['validation_AP']);p=np.load(selected.parent/'predictions.npy');pred[key]=p;thr=threshold(a['y'][va],p[va]);out['models'][key]={'selected':json.load(open(selected)),'test':metric(rows,a,p,te,thr),'quiet_test':metric(rows,a,p,te,thr,True)}
 # fixed test predictions and no future donor text: each document replaced with nearest earlier different-page report in preparation.
 for key in list(pred):
  path=R/'controls'/(key+'_mismatch.npy')
  if path.exists():
   p=np.load(path);base=out['models'][key]['test']['threshold'];out['models'][key+'_mismatch']={'test':metric(rows,a,p,te,base),'quiet_test':metric(rows,a,p,te,base,True)}
 np.savez_compressed(R/'all_predictions.npz',**pred);dump(R/'results_partial.json',out)
 print(json.dumps(out['models'],indent=2),flush=True)
 return out

def controls():
 torch.set_num_threads(2);rows,a=data();out=R/'controls';out.mkdir(exist_ok=True)
 for key in ['TFIDF','Qwen3','Mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct']:
  paths=list((R/'trials').glob(key+'_*/result.json'));assert len(paths)==12,(key,len(paths));selected=max(paths,key=lambda p:json.load(open(p))['validation_AP']);v=json.load(open(selected));f,x=features(key,rows,a);f[:,-32:]=np.load(D/(key+'_mismatch_features.npy'));m=Model(key,v['width'],f.shape[1]);m.load_state_dict(torch.load(selected.parent/'best.pt',map_location='cpu',weights_only=True));m.eval()
  with torch.inference_mode():p=np.concatenate([m(torch.tensor(x[ii]),torch.tensor(f[ii])).sigmoid().numpy() for ii in np.array_split(np.arange(len(rows)),max(1,int(np.ceil(len(rows)/1024))))])
  np.save(out/(key+'_mismatch.npy'),p)
 print('FIXED-WEIGHT MISMATCH COMPLETE',flush=True)
def finalize():
 out=report();rows,a=data();va=np.flatnonzero(a['split']=='val');te=np.flatnonzero(a['split']=='test');pred=dict(np.load(R/'all_predictions.npz'));assert len(pred)==12
 # Select numerical reference only on validation AP, never its test score.
 numkeys=['PatchTST','iTransformer','TimeMixer','Numeric','Chronos2'];ref=max(numkeys,key=lambda k:average_precision_score(a['y'][va],pred[k][va]));out['numerical_reference']=ref;out['paired_gains']={}
 for key in ['TFIDF','Qwen3','Mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct']:
  out['paired_gains'][key]={}
  for quiet in [False,True]:
   ii=te[a['quiet'][te]] if quiet else te;groups={(rows[i]['page'],int(a['first'][i])):[] for i in te if a['first'][i]>=0}
   for i in ii:
    if a['first'][i]<0:continue
    groups.setdefault((rows[i]['page'],int(a['first'][i])),[]).append(i)
   page=sorted({k[0] for k in groups});parts={p:[] for p in page};thr=out['models'][key]['test']['threshold'];rthr=out['models'][ref]['test']['threshold']
   for (p,onset),ids in groups.items():
    l1=max([onset-rows[i]['ordinal'] for i in ids if pred[key][i]>=thr],default=-1);l0=max([onset-rows[i]['ordinal'] for i in ids if pred[ref][i]>=rthr],default=-1);parts[p].append([int(l1>=0)-int(l0>=0),max(l1,0)-max(l0,0),int(l1>=0 and l0>=0),l1-l0 if l1>=0 and l0>=0 else 0])
   rng=np.random.default_rng(2024);est=np.concatenate(list(map(np.array,parts.values())),axis=0).mean(0);boot=[]
   for _ in range(2000):
    sampled=rng.choice(page,len(page),replace=True);boot.append(np.concatenate([parts[p] for p in sampled],axis=0).mean(0))
   ci=np.quantile(boot,[.025,.975],axis=0);out['paired_gains'][key]['quiet' if quiet else 'all']={'reference':ref,'event_recall_pp':float(100*est[0]),'recall_CI95_pp':(100*ci[:,0]).tolist(),'mean_lead_all_events_gain_days':float(est[1]),'lead_CI95_days':ci[:,1].tolist(),'common_detected_events':int(sum(sum(v[2] for v in z) for z in parts.values())),'lead_gain_common_detected_days':float(est[3]/est[2]) if est[2]>0 else None,'bootstrap':'2000 paired page clusters; validation-selected fixed models and thresholds'}
 # Full seven-origin opportunity and new-page diagnostics.
 trainpages={r['page'] for r in rows if r['split']=='train'};out['diagnostics']={'test_pages_unseen_in_train':len({rows[i]['page'] for i in te if rows[i]['page'] not in trainpages})}
 for key,p in pred.items():
  thr=out['models'][key]['test']['threshold'];new=np.array([i for i in te if rows[i]['page'] not in trainpages]);out['models'][key]['unseen_page_test']=metric(rows,a,p,new,thr)
 
 for key,p in pred.items():
  out['budgets'][key]={}
  for budget in [.001,.005,.01,.02,.05]:
   thr=threshold(a['y'][va],p[va],budget);out['budgets'][key][str(budget)]={'test':metric(rows,a,p,te,thr),'quiet_test':metric(rows,a,p,te,thr,True)}
  # Historical scale eligibility uses only observed values, not future event size.
  vv=va[np.median(a['x'][va],axis=1)>=100];tt=te[np.median(a['x'][te],axis=1)>=100];thr=threshold(a['y'][vv],p[vv]);out['models'][key]['observed_baseline_ge100']={'test':metric(rows,a,p,tt,thr),'quiet_test':metric(rows,a,p,tt,thr,True)}
 out['event_opportunity']={}
 for quiet in [False,True]:
  ii=te[a['quiet'][te]] if quiet else te;g={}
  for i in ii:
   if a['first'][i]>=0:g.setdefault((rows[i]['page'],int(a['first'][i])),set()).add(rows[i]['ordinal'])
  out['event_opportunity']['quiet' if quiet else 'all']={'events':len(g),'all7origins_available':sum(len(v)==7 for v in g.values()),'available_origins_histogram':{str(n):sum(len(v)==n for v in g.values()) for n in range(1,8)}}
 
 # Illustrations use validation-chosen LLM and first chronological qualifying cases, not best test family.
 llms=['Qwen3','Mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct'];lk=max(llms,key=lambda k:out['models'][k]['selected']['validation_AP']);lt=out['models'][lk]['test']['threshold'];rt=out['models'][ref]['test']['threshold'];docs=json.load(open(D/'documents.json'));di=json.load(open(D/'document_indices.json'));cases=[]
 for i in sorted(te,key=lambda i:(rows[i]['ordinal'],rows[i]['page'])):
  if not(a['quiet'][i] and a['y'][i] and pred[lk][i]>=lt and pred[ref][i]<rt):continue
  if any(c['page']==rows[i]['page'] and c['onset']==int(a['first'][i]) for c in cases):continue
  cases.append({'page':rows[i]['page'],'origin':rows[i]['origin'],'onset':int(a['first'][i]),'lead_days':int(a['first'][i]-rows[i]['ordinal']),'LLM':lk,'LLM_score':float(pred[lk][i]),'numeric_reference':ref,'numeric_score':float(pred[ref][i]),'last_observed_count':float(a['x'][i,-1]),'observed28median':float(np.median(a['x'][i])),'available_documents':[docs[j] for j in di[i]]})
  if len(cases)==3:break
 dump(R/'case_studies.json',{'selection':'LLM selected by validation AP; first chronological quiet positive origins detected by LLM and not validation-chosen numerical reference at respective fixed thresholds; illustrative not aggregate evidence','cases':cases});out['case_study_model']=lk;out['case_study_count']=len(cases)
 out['complete']=True;dump(R/'results.json',out)


 import csv
 with open(R/'comparison.csv','w') as f:
  writer=csv.writer(f);writer.writerow(['model','test_AP','test_FPR_percent','test_event_recall_percent','quiet_event_recall_percent','quiet_lead_all_events_days','quiet_lead_detected_days'])
  for k,v in out['models'].items():writer.writerow([k,v['test']['AP'],v['test']['FPR_percent'],v['test']['event_recall_percent'],v['quiet_test']['event_recall_percent'],v['quiet_test']['mean_lead_all_events_days'],v['quiet_test']['mean_lead_detected_days']])
 print('ALL ALERT EXPERIMENTS COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--key');p.add_argument('--width',type=int);p.add_argument('--lr',type=float);p.add_argument('--pw',type=int);args=p.parse_args()
 try:
  if args.stage=='worker':work(args.key,args.width,args.lr,args.pw)
  elif args.stage=='controls':controls()
  elif args.stage=='report':finalize()
 except Exception as e:
  dump(R/('failure_'+str(args.key)+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
