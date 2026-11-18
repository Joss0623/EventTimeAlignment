from run import read_npz
"""Second frozen LLM family under the identical rows, texts and readout search."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1')
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
from transformers import AutoModel,AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from run import load,snapshot_key,error,D,R,B,dump

def main():
 torch.set_num_threads(2);torch.manual_seed(2024);rows,x,y=load();tr=np.array([i for i,z in enumerate(rows) if z['split']=='train']);va=np.array([i for i,z in enumerate(rows) if z['split']=='val']);te=np.array([i for i,z in enumerate(rows) if z['split']=='test']);reps=[read_npz(D/'representations'/(snapshot_key(z)+'.npz')) for z in rows];texts=[str(z['text']) for z in reps];path=B/'pycodes/WebBurst/weights/mistralai--Mistral-7B-Instruct-v0.2';out=D/'mistral_embeddings.npz'
 if out.exists():q=np.load(out)['embedding']
 else:
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True);device_map={'embed_tokens':0,'norm':3,'rotary_emb':0,**{'layers.'+str(i):i//8 for i in range(32)}};m=AutoModel.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,device_map=device_map).eval();cache={};vecs=[];cache_dir=D/'mistral_representations';cache_dir.mkdir(exist_ok=True)
  for i,text in enumerate(texts):
   digest=hashlib.sha256(text.encode()).hexdigest()
   cached=cache_dir/(digest+'.npy')
   if digest not in cache and cached.exists():cache[digest]=np.load(cached)
   if digest not in cache:
    ids=tok(text,add_special_tokens=False)['input_ids'];parts=[ids[j:j+4096] for j in range(0,len(ids),4096)] or [[]];zs=[]
    for part in parts:
     t=torch.tensor([part],device='cuda')
     with torch.inference_mode():h=m(input_ids=t,use_cache=False).last_hidden_state;zs.append(h.float().mean(1).cpu().numpy()[0])
    cache[digest]=np.average(zs,axis=0,weights=[max(1,len(k)) for k in parts]).astype(np.float32)
    np.save(cached,cache[digest])
   vecs.append(cache[digest])
   if (i+1)%25==0:print('MISTRAL ENCODE',i+1,'/',len(texts),flush=True)
  q=np.stack(vecs);np.savez_compressed(out,embedding=q,ids=np.array([snapshot_key(z) for z in rows]));del m;torch.cuda.empty_cache()
 if '--embed-only' in sys.argv:return
 result=json.load(open(R/'results.json'));npz=np.load(R/'all_predictions.npz');pred={k:npz[k] for k in npz.files};base=pred[result['base']];nbase=np.maximum(np.median(x,1),1);lead=np.array([z['lead'] for z in rows]);f=np.c_[np.log1p(x)-np.log1p(nbase[:,None]),np.log1p(nbase),np.log1p(base)-np.log1p(nbase[:,None]),np.log1p(lead)];sc=StandardScaler().fit(f[tr]);f=sc.transform(f);target=np.log1p(y)-np.log1p(base)
 q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8);pc=PCA(n_components=16,random_state=2024).fit(q[tr]);q=pc.transform(q);sc=StandardScaler().fit(q[tr]);q=sc.transform(q)
 snap=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];import datetime as dt
 meta=np.array([[len(texts[i]),0 if not s['revision_time'] else (dt.datetime.fromisoformat(rows[i]['origin'].replace('Z','+00:00'))-dt.datetime.fromisoformat(s['revision_time'].replace('Z','+00:00'))).total_seconds()/86400,int(s['available'])] for i,s in enumerate(snap)]);sc=StandardScaler().fit(meta[tr]);meta=sc.transform(meta);best=np.inf;trials=[]
 for w in [.1,1.,3.]:
  design=np.c_[f,w*np.c_[meta,q]]
  for alpha in [1.,10.,100.,1000.]:
   model=Ridge(alpha=alpha).fit(design[tr],target[tr]);p=np.maximum(np.expm1(np.clip(np.log1p(base)+model.predict(design),0,25)),0);v=float(error(y[va],p[va]).mean());trials.append({'weight':w,'alpha':alpha,'val_WAPE':v})
   if v<best:best=v;chosen={'weight':w,'alpha':alpha,'val_WAPE':v};bestmodel=model;pred['Mistral7B']=p
 donors=[]
 for i,z in enumerate(rows):
  pool=[j for j,w in enumerate(rows) if w['pageid']!=z['pageid'] and snap[j]['available'] and snap[j]['revision_time']<z['origin'] and w['origin']<=z['origin']];donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else -1)
 wrong=np.array([q[j] if j>=0 else np.zeros(q.shape[1]) for j in donors]);freeze=[next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]
 for name,qq in [('Mistral7BFixedMismatch',wrong),('Mistral7BFrozen7d',q[freeze])]:pred[name]=np.maximum(np.expm1(np.clip(np.log1p(base)+bestmodel.predict(np.c_[f,chosen['weight']*np.c_[meta,qq]]),0,25)),0)
 groups=np.array([z['pageid'] for z in rows])
 for kind in ['event','calendar_control','all']:
  for l in [7,3,1]:
   idx=np.array([i for i in te if rows[i]['lead']==l and (kind=='all' or rows[i]['kind']==kind)]);gg=np.unique(groups[idx]);rng=np.random.default_rng(2024);samples=[np.concatenate([np.flatnonzero(groups[idx]==g) for g in rng.choice(gg,len(gg),replace=True)]) for _ in range(2000)]
   for name in ['Mistral7B','Mistral7BFixedMismatch','Mistral7BFrozen7d']:
    p=pred[name];e=error(y[idx],p[idx]);result['curves'][kind][str(l)][name]={'WAPE':float(e.mean()),'CI95':np.quantile([e[j].mean() for j in samples],[.025,.975]).tolist(),'RMSLE':float(np.sqrt(np.mean((np.log1p(y[idx])-np.log1p(p[idx]))**2))),'n':len(idx)}
   for first in ['Chronos2','NumericReadout','Metadata','TFIDF','Mistral7BFixedMismatch','Mistral7BFrozen7d']:
    e1=error(y[idx],pred[first][idx]);e2=error(y[idx],pred['Mistral7B'][idx]);diff=e1-e2;result['gains'][kind][str(l)][first+' -> Mistral7B']={'pp':float(diff.mean()),'relative_percent':float(100*diff.mean()/max(e1.mean(),1e-8)),'CI95':np.quantile([diff[j].mean() for j in samples],[.025,.975]).tolist()}
 np.savez_compressed(R/'all_predictions.npz',**pred);dump(R/'results.json',result);dump(R/'mistral_selection.json',{'model':str(path),'frozen':True,'same_input_strings_as_Qwen':True,'pooling':'token mean; full common input retained,4096-token chunk length-weighted pooling if needed','selected':chosen,'trials':trials,'seed':2024});print('MISTRAL COMPLETE',chosen,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback;dump(R/'mistral_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
