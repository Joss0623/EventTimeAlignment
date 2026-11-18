from run import read_npz
"""Bounded validation tuning: equal readout search and fixed-weight controls."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import json,hashlib,datetime as dt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from run import load,snapshot_key,error,D,R,dump
T=R/'tuning01';T.mkdir(exist_ok=True)
protocol={'seed':2024,'alpha':[100,1000,10000,100000,1000000,10000000],'text_weight':[.1,1,3,10],'pca_dimensions':[4,8,16,32],'residual_strength':[0,.25,.5,.75,1],'primary_selection':'pooled validation macro WAPE across event/calendar and leads','secondary_selection':'event-validation macro WAPE, explicitly event-conditioned diagnostic','all_models_same_readout_budget':True,'test_used_for_selection':False,'frozen_feature_extraction':True}
dump(T/'protocol.json',protocol)
rows,x,y=load();tr=np.array([i for i,z in enumerate(rows) if z['split']=='train']);va=np.array([i for i,z in enumerate(rows) if z['split']=='val']);te=np.array([i for i,z in enumerate(rows) if z['split']=='test']);ve=np.array([i for i in va if rows[i]['kind']=='event']);num=np.load(R/'numeric_predictions.npz');base=num['Chronos2'];med=np.maximum(np.median(x,1),1);lead=np.array([z['lead'] for z in rows]);f=np.c_[np.log1p(x)-np.log1p(med[:,None]),np.log1p(med),np.log1p(base)-np.log1p(med[:,None]),np.log1p(lead)];f=StandardScaler().fit(f[tr]).transform(f);target=np.log1p(y)-np.log1p(base)
reps=[read_npz(D/'representations'/(snapshot_key(z)+'.npz')) for z in rows];texts=[str(v['text']) for v in reps];snap=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];meta=np.array([[len(texts[i]),0 if not s['revision_time'] else (dt.datetime.fromisoformat(rows[i]['origin'].replace('Z','+00:00'))-dt.datetime.fromisoformat(s['revision_time'].replace('Z','+00:00'))).total_seconds()/86400,int(s['available'])] for i,s in enumerate(snap)]);meta=StandardScaler().fit(meta[tr]).transform(meta)
raw={'Qwen3':np.stack([v['embedding'] for v in reps])}
for k,fn in [('Mistral7B','mistral'),('GPToss20B','gpt-oss-20b'),('DeepSeekV2Lite','deepseek-v2-lite-chat'),('KimiVLA3B','kimi-vl-a3b-instruct')]:raw[k]=np.load(D/(fn+'_embeddings.npz'))['embedding']
tf=TfidfVectorizer(min_df=2,max_features=2000,ngram_range=(1,2),sublinear_tf=True).fit([texts[i] for i in tr]);lex=tf.transform(texts).toarray();lex*=np.sqrt(16)/max(np.linalg.norm(lex[tr],axis=1).mean(),1e-8)
donors=[]
for i,z in enumerate(rows):
 pool=[j for j,w in enumerate(rows) if w['pageid']!=z['pageid'] and snap[j]['available'] and snap[j]['revision_time']<z['origin'] and w['origin']<=z['origin']]
 donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else -1)
freeze=[next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]
allpred={'Chronos2':base};selected={};alltrials={};objectives={'pooled':va,'event':ve}
for name in ['NumericReadout','Metadata','TFIDF']+list(raw):
 variants={0:None} if name=='NumericReadout' else {0:meta} if name=='Metadata' else {0:np.c_[meta,lex]} if name=='TFIDF' else {}
 if name in raw:
  q=raw[name]/np.maximum(np.linalg.norm(raw[name],axis=1,keepdims=True),1e-8)
  for dim in protocol['pca_dimensions']:
   pp=PCA(n_components=min(dim,len(tr)-1),random_state=2024).fit(q[tr]);a=pp.transform(q);a=StandardScaler().fit(a[tr]).transform(a);variants[dim]=np.c_[meta,a]
 best={o:np.inf for o in objectives};models={};trials=[]
 for dim,extra in variants.items():
  for weight in ([1] if extra is None else protocol['text_weight']):
   design=f if extra is None else np.c_[f,weight*extra]
   for alpha in protocol['alpha']:
    m=Ridge(alpha=alpha).fit(design[tr],target[tr]);delta=m.predict(design)
    for beta in protocol['residual_strength']:
     p=np.maximum(np.expm1(np.clip(np.log1p(base)+beta*delta,0,25)),0);cfg={'alpha':alpha,'weight':weight,'pca_dim':dim,'residual_strength':beta};cfg.update({o+'_val_WAPE':float(error(y[ii],p[ii]).mean()) for o,ii in objectives.items()});trials.append(cfg)
     for o in objectives:
      v=cfg[o+'_val_WAPE']
      if v<best[o]-1e-10:best[o]=v;selected[name+'_'+o]=dict(cfg);allpred[name+'_'+o]=p;models[o]=(m,extra,weight,beta)
 for o,(m,extra,weight,beta) in models.items():
  if name in raw:
   qq=extra[:,3:];wrong=np.array([qq[j] if j>=0 else np.zeros(qq.shape[1]) for j in donors])
   for suffix,aa in [('FixedMismatch',wrong),('Frozen7d',qq[freeze]),('ZeroContent',np.zeros_like(qq))]:
    design=np.c_[f,weight*np.c_[meta,aa]];allpred[name+suffix+'_'+o]=np.maximum(np.expm1(np.clip(np.log1p(base)+beta*m.predict(design),0,25)),0)
 alltrials[name]=trials;print('SELECTED',name,{o:selected[name+'_'+o] for o in objectives},flush=True);dump(T/'selection.json',selected);dump(T/'trials.json',alltrials)
np.savez_compressed(T/'predictions.npz',**allpred)
result={'seed':2024,'curves':{},'gains':{},'selection':selected,'n_pages':len({z['pageid'] for z in rows}),'n_test_pages':len({rows[i]['pageid'] for i in te})}
fresh_pages=set(json.load(open(R/'protocol.json')).get('fresh_test_pageids',[])) if (R/'protocol.json').exists() else set()
kinds=['event','calendar_control','all']+(['fresh_event','fresh_calendar_control'] if fresh_pages else [])
for kind in kinds:
 result['curves'][kind]={};result['gains'][kind]={}
 for l in [7,3,1]:
  ii=np.array([i for i in te if rows[i]['lead']==l and (kind=='all' or (kind.startswith('fresh_') and rows[i]['pageid'] in fresh_pages and rows[i]['kind']==kind[6:]) or rows[i]['kind']==kind)]);groups=np.array([rows[i]['pageid'] for i in ii]);gg=np.unique(groups);rng=np.random.default_rng(2024);boot=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(gg,len(gg),replace=True)]) for _ in range(2000)];es={k:error(y[ii],v[ii]) for k,v in allpred.items()};result['curves'][kind][str(l)]={k:{'WAPE':float(a.mean()),'RMSLE':float(np.sqrt(np.mean((np.log1p(y[ii])-np.log1p(allpred[k][ii]))**2))),'n':len(ii)} for k,a in es.items()};result['gains'][kind][str(l)]={}
  for o in objectives:
   for k in raw:
    for first in ['NumericReadout_'+o,k+'FixedMismatch_'+o,k+'Frozen7d_'+o,k+'ZeroContent_'+o]:
     second=k+'_'+o;dd=es[first]-es[second];result['gains'][kind][str(l)][first+' -> '+second]={'pp':float(dd.mean()),'CI95':np.quantile([dd[j].mean() for j in boot],[.025,.975]).tolist()}
dump(T/'results.json',result);print('COMPLETE',flush=True)

from case_study import main as run_cases
run_cases()
