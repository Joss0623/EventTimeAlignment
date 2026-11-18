"""Point-in-time fixed-target study: pretrained numerical and text encoders, matched readouts."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import json,time,re,argparse,hashlib,random,copy
import numpy as np
B=Path('/home/home_new/qsmx');D=Path(os.environ.get('ATTENTION_PIT_DATA_ROOT',str(B/'data/AttentionPIT/pilot01')));R=Path(os.environ.get('ATTENTION_PIT_RUN_ROOT',str(B/'runs/AttentionPIT/pilot01')));R.mkdir(parents=True,exist_ok=True)
def dump(p,z):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(z,ensure_ascii=False,indent=2));q.replace(p)
def read_npz(path):
 with np.load(path) as f:return {k:f[k].copy() for k in f.files}
def load():
 c=json.load(open(D/'cohort.json'));rows=[z for z in c['rows'] if z['numeric_complete']];return rows,np.array([z['history28'] for z in rows],np.float32),np.array([z['target7'] for z in rows],np.float32)
def error(y,p):return np.abs(y-p).sum(1)/np.maximum(y.sum(1),1)*100

def numerical():
 import torch
 from chronos import BaseChronosPipeline
 from sklearn.linear_model import Ridge
 from sklearn.preprocessing import StandardScaler
 rows,x,y=load();tr=np.array([i for i,z in enumerate(rows) if z['split']=='train' and z['numeric_complete']]);va=np.array([i for i,z in enumerate(rows) if z['split']=='val' and z['numeric_complete']]);assert np.isfinite(x).all() and np.isfinite(y).all()
 torch.set_num_threads(2);torch.manual_seed(2024);out={};base=np.median(x,axis=1);leads=np.array([z['lead'] for z in rows]);f=np.c_[np.log1p(x)-np.log1p(base[:,None]),np.log1p(base),np.log1p(leads)];sc=StandardScaler().fit(f[tr]);f=sc.transform(f);target=np.log1p(y)-np.log1p(base[:,None]);trials=[]
 for alpha in [.1,1,10,100,1000]:
  m=Ridge(alpha=alpha).fit(f[tr],target[tr]);p=np.maximum(np.expm1(np.clip(m.predict(f)+np.log1p(base[:,None]),0,25)),0);v=float(error(y[va],p[va]).mean());trials.append({'alpha':alpha,'val_WAPE':v})
  if len(trials)==1 or v<best:best=v;out['Ridge']=p;selected=alpha
 out['Persistence']=np.repeat(x[:,-1:],7,axis=1);out['SeasonalNaive']=np.stack([x[i,21+(int(leads[i])+np.arange(7))%7] for i in range(len(rows))])
 pipe=BaseChronosPipeline.from_pretrained(str(B/'pycodes/WebBurst/weights/amazon--chronos-2'),device_map='cuda',torch_dtype=torch.bfloat16,local_files_only=True);qi=pipe.quantiles.index(.5);cp={}
 for mode in ['raw','log']:
  p=np.zeros_like(y)
  for lead in [7,3,1]:
   ids=np.flatnonzero(leads==lead);v=np.log1p(x[ids]) if mode=='log' else x[ids];z=pipe.predict(torch.tensor(v).unsqueeze(1),prediction_length=lead+7,batch_size=16,context_length=28,cross_learning=False);ar=np.stack([s[0,qi,-7:].cpu().float().numpy() for s in z]);p[ids]=np.maximum(np.expm1(np.clip(ar,0,25)) if mode=='log' else ar,0)
  cp[mode]=p;print('CHRONOS',mode,float(error(y[va],p[va]).mean()),flush=True)
 mode=min(cp,key=lambda k:error(y[va],cp[k][va]).mean());out['Chronos2']=cp[mode];np.savez_compressed(R/'numeric_predictions.npz',**out);dump(R/'numeric_selection.json',{'Ridge':{'selected_alpha':selected,'trials':trials},'Chronos2':{'selected_mode':mode,'val_scores':{k:float(error(y[va],v[va]).mean()) for k,v in cp.items()}},'complete':True});print('NUMERIC COMPLETE',flush=True)

def snapshot_key(z):return hashlib.sha256((str(z['pageid'])+'|'+z['origin']).encode()).hexdigest()[:24]
def plain(text):
 text=re.sub(r'<!--.*?-->',' ',text,flags=re.S);text=re.sub(r'<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>',' ',text,flags=re.S|re.I)
 # Retain source text without expanding templates, external pages or present-day content.
 for _ in range(12):text=re.sub(r'\{\{[^{}]*\}\}',' ',text)
 text=re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]',r'\1',text);text=re.sub(r'\[https?://\S+\s*([^\]]*)\]',r'\1',text);text=re.sub(r'<[^>]+>',' ',text);return re.sub(r'\s+',' ',text).strip()
def embed():
 import torch
 from transformers import AutoModel,AutoTokenizer
 torch.set_num_threads(2);torch.manual_seed(2024);rows,_,_=load();modelpath=B/'models/ChronoGeom/qwen3-8b-b968826d';tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True);tok.padding_side='left';m=AutoModel.from_pretrained(modelpath,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();outdir=D/'representations';outdir.mkdir(exist_ok=True);done={};cache={}
 while True:
  new=0
  for z in rows:
   key=snapshot_key(z);dest=outdir/(key+'.npz');snap=D/'snapshots'/(key+'.json')
   if dest.exists() or not snap.exists():continue
   v=json.load(open(snap));assert not v['available'] or v['revision_time']<z['origin'];content=plain(v['wikitext'] or 'No historical article text was available.');ids=tok(content,add_special_tokens=False)['input_ids'];retained=ids if len(ids)<=2048 else ids[:1536]+ids[-512:];content=tok.decode(retained,skip_special_tokens=True);digest=hashlib.sha256(content.encode()).hexdigest()
   if digest in cache:vec=cache[digest]
   else:
    t=tok(content,return_tensors='pt').to('cuda')
    with torch.inference_mode():h=m(**t,use_cache=False).last_hidden_state.float();mask=t.attention_mask[:,:,None];vec=((h*mask).sum(1)/mask.sum(1)).cpu().numpy()[0]
    cache[digest]=vec
   np.savez_compressed(dest,embedding=vec.astype(np.float32),text=np.array(content));done[key]=True;new+=1;print('EMBED',len(list(outdir.glob('*.npz'))),'/',len(rows),flush=True)
  if all((outdir/(snapshot_key(z)+'.npz')).exists() for z in rows):break
  time.sleep(10)
 dump(R/'embedding_source.json',{'model':str(modelpath),'frozen':True,'generation':False,'pooling':'masked mean last hidden layer','max_tokens':2048,'selection':'first1536 + last512 when longer; lexical and Qwen conditions use identical decoded input','unique_inputs_encoded':len(cache),'rows':len(rows),'asof_revision_checked':True});print('EMBED COMPLETE',flush=True)

def analyze():
 from sklearn.preprocessing import StandardScaler,OneHotEncoder
 from sklearn.linear_model import Ridge
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.decomposition import PCA
 rows,x,y=load();outdir=D/'representations'
 while not (R/'numeric_predictions.npz').exists() or not all((outdir/(snapshot_key(z)+'.npz')).exists() for z in rows):time.sleep(10)
 tr=np.array([i for i,z in enumerate(rows) if z['split']=='train']);va=np.array([i for i,z in enumerate(rows) if z['split']=='val']);te=np.array([i for i,z in enumerate(rows) if z['split']=='test']);num=np.load(R/'numeric_predictions.npz');pred={k:num[k] for k in num.files};basekey=min(pred,key=lambda k:error(y[va],pred[k][va]).mean());base=pred[basekey]
 nbase=np.maximum(np.median(x,axis=1),1);lead=np.array([z['lead'] for z in rows]);f=np.c_[np.log1p(x)-np.log1p(nbase[:,None]),np.log1p(nbase),np.log1p(base)-np.log1p(nbase[:,None]),np.log1p(lead)];fs=StandardScaler().fit(f[tr]);f=fs.transform(f);target=np.log1p(y)-np.log1p(base)
 arrays=[read_npz(outdir/(snapshot_key(z)+'.npz')) for z in rows];texts=[str(v['text']) for v in arrays];q=np.stack([v['embedding'] for v in arrays]);q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8);pc=PCA(n_components=16,random_state=2024).fit(q[tr]);q=pc.transform(q);qs=StandardScaler().fit(q[tr]);q=qs.transform(q)
 tf=TfidfVectorizer(min_df=2,max_features=2000,ngram_range=(1,2),sublinear_tf=True).fit([texts[i] for i in tr]);lex=tf.transform(texts).toarray();ln=np.linalg.norm(lex[tr],axis=1).mean();lex*=np.sqrt(16)/max(ln,1e-8)
 # Information-volume/age control; no content words or identity in metadata.
 snap=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];import datetime as dt
 meta=np.array([[len(texts[i]),0 if not s['revision_time'] else (dt.datetime.fromisoformat(rows[i]['origin'].replace('Z','+00:00'))-dt.datetime.fromisoformat(s['revision_time'].replace('Z','+00:00'))).total_seconds()/86400,int(s['available'])] for i,s in enumerate(snap)]);ms=StandardScaler().fit(meta[tr]);meta=ms.transform(meta)
 donors=[]
 for i,z in enumerate(rows):
  pool=[j for j,w in enumerate(rows) if w['pageid']!=z['pageid'] and snap[j]['available'] and snap[j]['revision_time']<z['origin'] and w['origin']<=z['origin']]
  donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else -1)
 wrong=np.array([q[j] if j>=0 else np.zeros(q.shape[1]) for j in donors]);alltrials={};models={}
 for name,extra in [('NumericReadout',None),('Metadata',meta),('TFIDF',np.c_[meta,lex]),('Qwen3',np.c_[meta,q]),('Qwen3Mismatch',np.c_[meta,wrong])]:
  best=np.inf;ts=[]
  for weight in ([1.] if extra is None else [.1,1.,3.]):
   design=f if extra is None else np.c_[f,weight*extra]
   for alpha in [1.,10.,100.,1000.]:
    model=Ridge(alpha=alpha).fit(design[tr],target[tr]);p=np.maximum(np.expm1(np.clip(np.log1p(base)+model.predict(design),0,25)),0);v=float(error(y[va],p[va]).mean());ts.append({'alpha':alpha,'weight':weight,'val_WAPE':v})
    if v<best:best=v;pred[name]=p;models[name]={'alpha':alpha,'weight':weight,'val_WAPE':v}
  alltrials[name]=ts
 dump(R/'selection.json',{'base':basekey,'selected':models,'trials':alltrials,'fit':'closed-form ridge to optimum; all preprocessing fit only on train; pooled-lead validation WAPE; seed2024','split_groups':{'train':len(set(rows[i]['pageid'] for i in tr)),'val':len(set(rows[i]['pageid'] for i in va)),'test':len(set(rows[i]['pageid'] for i in te))}})
 # Fixed Qwen weights, wrong words only; recipient metadata unchanged.
 c=models['Qwen3'];design=np.c_[f,c['weight']*np.c_[meta,q]];m=Ridge(alpha=c['alpha']).fit(design[tr],target[tr]);pred['Qwen3FixedMismatch']=np.maximum(np.expm1(np.clip(np.log1p(base)+m.predict(np.c_[f,c['weight']*np.c_[meta,wrong]]),0,25)),0)
 frozen_idx=[next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]
 pred['Qwen3Frozen7d']=np.maximum(np.expm1(np.clip(np.log1p(base)+m.predict(np.c_[f,c['weight']*np.c_[meta,q[frozen_idx]]]),0,25)),0)
 np.savez_compressed(R/'all_predictions.npz',**pred);groups=np.array([z['pageid'] for z in rows]);result={'seed':2024,'base':basekey,'n_pages':40,'n_test_pages':12,'curves':{},'gains':{},'snapshot_stats':{}}
 for kind in ['event','calendar_control','all']:
  result['curves'][kind]={};result['gains'][kind]={}
  for l in [7,3,1]:
   idx=np.array([i for i in te if rows[i]['lead']==l and (kind=='all' or rows[i]['kind']==kind)]);gg=np.unique(groups[idx]);rng=np.random.default_rng(2024);samples=[np.concatenate([np.flatnonzero(groups[idx]==g) for g in rng.choice(gg,len(gg),replace=True)]) for _ in range(2000)]
   result['curves'][kind][str(l)]={}
   for name,p in pred.items():
    e=error(y[idx],p[idx]);boot=[e[j].mean() for j in samples];result['curves'][kind][str(l)][name]={'WAPE':float(e.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist(),'RMSLE':float(np.sqrt(np.mean((np.log1p(y[idx])-np.log1p(p[idx]))**2))),'n':len(idx)}
   result['gains'][kind][str(l)]={}
   for first,second in [('Chronos2','Qwen3'),('NumericReadout','Qwen3'),('Qwen3Frozen7d','Qwen3'),('Metadata','Qwen3'),('TFIDF','Qwen3'),('Qwen3Mismatch','Qwen3'),('Qwen3FixedMismatch','Qwen3')]:
    e1=error(y[idx],pred[first][idx]);e2=error(y[idx],pred[second][idx]);d=e1-e2;boot=[d[j].mean() for j in samples];result['gains'][kind][str(l)][first+' -> '+second]={'pp':float(d.mean()),'relative_percent':float(100*d.mean()/max(e1.mean(),1e-8)),'CI95':np.quantile(boot,[.025,.975]).tolist()}
 for kind in ['event','calendar_control']:
  ids=[i for i,z in enumerate(rows) if z['kind']==kind];changed=0
  for event in {rows[i]['event_id'] for i in ids}:
   ii=[i for i in ids if rows[i]['event_id']==event];changed+=len({snap[i]['revid'] for i in ii})>1
  result['snapshot_stats'][kind]={'rows':len(ids),'available':sum(snap[i]['available'] for i in ids),'events_with_changed_revision':changed,'events_with_changed_model_text':sum(len({texts[i] for i in ids if rows[i]['event_id']==event})>1 for event in {rows[i]['event_id'] for i in ids}),'events':len({rows[i]['event_id'] for i in ids})}
 dump(R/'results.json',result);print('ALL COMPLETE',json.dumps(result['curves']['event']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['numerical','embed','analyze']);a=p.parse_args()
 try:globals()[a.stage]()
 except Exception as e:
  import traceback;dump(R/(a.stage+'_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
