from run import read_npz
"""Matched neural readouts: numeric/metadata/lexical/five frozen LLMs, full training."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import sys,time,json,argparse,itertools,subprocess,traceback,hashlib,datetime as dt
from pathlib import Path
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from run import load,error,D,R,B,dump,snapshot_key
O=R/'neural_readouts';O.mkdir(exist_ok=True);C=B/'pycodes/AttentionPIT';KEYS=['NumericReadout','Metadata','TFIDF','Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B'];LLMS=KEYS[3:]
def prepare():
 rows,x,y=load();tr=np.array([i for i,z in enumerate(rows) if z['split']=='train']);va=np.array([i for i,z in enumerate(rows) if z['split']=='val']);n=np.load(R/'numeric_predictions.npz');dd=np.load(R/'deep_baselines/predictions.npz');candidates={'Chronos2':n['Chronos2'],**{k:dd[k] for k in dd.files}};best=min(candidates,key=lambda k:error(y[va],candidates[k][va]).mean());base=candidates[best];med=np.maximum(np.median(x,1),1);f=np.c_[np.log1p(x)-np.log1p(med[:,None]),np.log1p(med),np.log1p(base)-np.log1p(med[:,None]),np.log1p([z['lead'] for z in rows])];f=StandardScaler().fit(f[tr]).transform(f)
 reps=[read_npz(D/'representations'/(snapshot_key(z)+'.npz')) for z in rows];texts=[str(v['text']) for v in reps];snap=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];meta=np.array([[len(texts[i]),0 if not z['revision_time'] else (dt.datetime.fromisoformat(rows[i]['origin'].replace('Z','+00:00'))-dt.datetime.fromisoformat(z['revision_time'].replace('Z','+00:00'))).total_seconds()/86400,int(z['available'])] for i,z in enumerate(snap)]);meta=StandardScaler().fit(meta[tr]).transform(meta)
 raw={'Qwen3':np.stack([v['embedding'] for v in reps])}
 for k,fn in [('Mistral7B','mistral'),('GPToss20B','gpt-oss-20b'),('DeepSeekV2Lite','deepseek-v2-lite-chat'),('KimiVLA3B','kimi-vl-a3b-instruct')]:raw[k]=np.load(D/(fn+'_embeddings.npz'))['embedding']
 tf=TfidfVectorizer(min_df=2,max_features=2000,ngram_range=(1,2),sublinear_tf=True).fit([texts[i] for i in tr]);lex=tf.transform(texts);features={'NumericReadout_0':np.c_[f,np.zeros((len(rows),35))],'Metadata_0':np.c_[f,meta,np.zeros((len(rows),32))]};donors=[]
 for i,z in enumerate(rows):
  pool=[j for j,w in enumerate(rows) if w['pageid']!=z['pageid'] and snap[j]['available'] and snap[j]['revision_time']<z['origin'] and w['origin']<=z['origin']];donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else -1)
 frozen=[next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]
 for k in ['TFIDF']+LLMS:
  for dim in [8,32]:
   if k=='TFIDF':project=TruncatedSVD(n_components=dim,random_state=2024).fit(lex[tr]);q=project.transform(lex)
   else:
    a=raw[k]/np.maximum(np.linalg.norm(raw[k],axis=1,keepdims=True),1e-8);project=PCA(n_components=dim,random_state=2024).fit(a[tr]);q=project.transform(a)
   q=StandardScaler().fit(q[tr]).transform(q);q=np.pad(q,((0,0),(0,32-dim)));features[k+'_'+str(dim)]=np.c_[f,meta,q]
   if k in LLMS:
    wrong=np.array([q[j] if j>=0 else np.zeros(32) for j in donors])
    for suffix,qq in [('FixedMismatch',wrong),('Frozen7d',q[frozen]),('ZeroContent',np.zeros_like(q))]:features[k+suffix+'_'+str(dim)]=np.c_[f,meta,qq]
 np.savez_compressed(O/'features.npz',**{k:v.astype(np.float32) for k,v in features.items()},target=(np.log1p(y)-np.log1p(base)).astype(np.float32),base=base,y=y,tr=tr,va=va);dump(O/'feature_protocol.json',{'numeric_reference':best,'reference_validation_WAPE':{k:float(error(y[va],v[va]).mean()) for k,v in candidates.items()},'frozen_encoders':True,'pca_train_only':[8,32],'same_head':'Linear-GELU-Dropout-Linear-GELU-Linear7','same_information':True});print('FEATURES READY',best,flush=True)
def net(dim,width,drop):return torch.nn.Sequential(torch.nn.Linear(dim,width),torch.nn.GELU(),torch.nn.Dropout(drop),torch.nn.Linear(width,width),torch.nn.GELU(),torch.nn.Linear(width,7))
def worker(a):
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024);name=f'{a.condition}_d{a.dim}_w{a.width}_lr{a.lr:g}_{a.loss}';dest=O/name;dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():return
 global FEATURE_DATA
 if 'FEATURE_DATA' not in globals():FEATURE_DATA=read_npz(O/'features.npz')
 d=FEATURE_DATA;xx=torch.tensor(d[a.condition+'_'+str(a.dim)],device='cuda');tt=torch.tensor(d['target'],device='cuda');tr=d['tr'];va=d['va'];base=d['base'];y=d['y'];m=net(xx.shape[1],a.width,.1).cuda();opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);rng=np.random.default_rng(2024);best=np.inf;stale=0;curve=[];be=0
 for epoch in range(1,151):
  m.train();losses=[]
  for ids in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/64)))):
   opt.zero_grad(set_to_none=True);p=m(xx[ids]);loss=torch.nn.functional.mse_loss(p,tt[ids]) if a.loss=='logMSE' else torch.nn.functional.smooth_l1_loss(p,tt[ids]);assert torch.isfinite(loss),name;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach()))
  m.eval()
  with torch.inference_mode():pv=np.expm1(np.clip(np.log1p(base[va])+m(xx[va]).cpu().numpy(),0,25));score=float(error(y[va],pv).mean())
  curve.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_WAPE':score})
  if score<best-1e-8:best=score;be=epoch;stale=0;torch.save(m.state_dict(),dest/'best.pt')
  else:stale+=1
  if epoch%10==0:dump(dest/'curve.json',curve)
  if epoch>=30 and stale>=20:break
 dump(dest/'curve.json',curve);dump(dest/'result.json',{'condition':a.condition,'dim':a.dim,'width':a.width,'lr':a.lr,'loss':a.loss,'seed':2024,'epochs_trained':epoch,'best_epoch':be,'validation_WAPE':best,'min_epochs':30,'max_epochs':150,'patience':20});print('COMPLETE',name,be,best,flush=True)
def report():
 torch.set_num_threads(2);rows,x,y=load();d=np.load(O/'features.npz');pred={'NumericalReference':d['base']};base=d['base'];selected={};compat={};pcompat={};dd=np.load(R/'deep_baselines/predictions.npz');pred.update({k:dd[k] for k in dd.files});pred['Chronos2']=np.load(R/'numeric_predictions.npz')['Chronos2']
 for k in KEYS:
  files=list(O.glob(k+'_d*/result.json'));assert len(files)==24,(k,len(files));f=min(files,key=lambda f:json.load(open(f))['validation_WAPE']);v=json.load(open(f));design=d[k+'_'+str(v['dim'])];m=net(design.shape[1],v['width'],.1);m.load_state_dict(torch.load(f.parent/'best.pt',map_location='cpu',weights_only=True));m.eval();selected[k]=v;compat[k+'_pooled']={'pooled_val_WAPE':v['validation_WAPE']}
  with torch.inference_mode():
   for name in [k]+([k+s for s in ['FixedMismatch','Frozen7d','ZeroContent']] if k in LLMS else []):pv=m(torch.tensor(d[name+'_'+str(v['dim'])])).numpy();p=np.maximum(np.expm1(np.clip(np.log1p(base)+pv,0,25)),0);pred[name]=p;pcompat[name+'_pooled']=p
 np.savez_compressed(O/'all_predictions.npz',**pred);np.savez_compressed(O/'predictions.npz',**pcompat);dump(O/'selection.json',compat);dump(O/'selected_configs.json',selected);fresh=set(json.load(open(R/'protocol.json'))['fresh_test_pageids']);res={'seed':2024,'n_pages':len({z['pageid'] for z in rows}),'curves':{},'gains':{},'selection':selected,'numeric_reference':json.load(open(O/'feature_protocol.json'))['numeric_reference']}
 for kind in ['event','calendar_control','fresh_event','fresh_calendar_control']:
  res['curves'][kind]={};res['gains'][kind]={}
  for l in [7,3,1]:
   ii=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['lead']==l and (z['kind']==kind or kind.startswith('fresh_') and z['kind']==kind[6:] and z['pageid'] in fresh)]);gg=np.array([rows[i]['pageid'] for i in ii]);unique=np.unique(gg);rng=np.random.default_rng(2024);boots=[np.concatenate([np.flatnonzero(gg==g) for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(2000)];es={k:error(y[ii],p[ii]) for k,p in pred.items()};res['curves'][kind][str(l)]={k:{'WAPE':float(v.mean()),'RMSLE':float(np.sqrt(np.mean((np.log1p(y[ii])-np.log1p(pred[k][ii]))**2))),'n':len(ii)} for k,v in es.items()};res['gains'][kind][str(l)]={}
   for k in LLMS:
    for first in ['NumericalReference','Chronos2','PatchTST','iTransformer','TimeMixer','NumericReadout','Metadata','TFIDF',k+'FixedMismatch',k+'Frozen7d',k+'ZeroContent']:
     delta=es[first]-es[k];res['gains'][kind][str(l)][first+' -> '+k]={'pp':float(delta.mean()),'CI95':np.quantile([delta[j].mean() for j in boots],[.025,.975]).tolist()}
 dump(O/'results.json',res)
 # Reuse the same explicit case-selection policy and single-column figure.
 os.environ['ATTENTION_PIT_CASE_SUBDIR']='neural_readouts'
 from case_study import main
 main();dump(O/'state.json',{'complete':True,'configurations_completed':192,'seed':2024});print('ALL NEURAL READOUTS COMPLETE',flush=True)
def manage():
 dump(O/'protocol.json',{'seed':2024,'conditions':KEYS,'configurations_per_condition':24,'lr':[.0001,.0003,.001],'loss':['logMSE','logHuber'],'width_numeric_metadata':[32,64,128,256],'width_text':[64,128],'PCA_text':[8,32],'min_epochs':30,'max_epochs':150,'patience':20,'batch':64,'selection':'same pooled validation WAPE for every condition, complete configuration training first; no test selection','controls':['fixed weights mismatched words','frozen7day words','zero content'],'main_numeric_candidates':['Chronos2','PatchTST','iTransformer','TimeMixer']})
 needed=[D/'ready.json',R/'embedding_source.json',R/'deep_baselines/results.json']+[D/(k+'_embeddings.npz') for k in ['mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct']]
 while not all(p.exists() for p in needed):
  dump(O/'state.json',{'stage':'waiting_for_common_features','missing':[str(p.name) for p in needed if not p.exists()]});time.sleep(20)
 if not (O/'features.npz').exists():prepare()
 queue=[]
 for k in KEYS:
  for dim,w,lr,loss in itertools.product([0] if k in KEYS[:2] else [8,32],[32,64,128,256] if k in KEYS[:2] else [64,128],[.0001,.0003,.001],['logMSE','logHuber']):queue.append((k,dim,w,lr,loss))
 free={int(v.split(',')[0]):int(v.split(',')[1]) for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
 gpus=[g for g,v in free.items() if v>=3000]
 while not gpus:
  time.sleep(20);free={int(v.split(',')[0]):int(v.split(',')[1]) for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};gpus=[g for g,v in free.items() if v>=3000]
 active={}
 for j,gpu in enumerate(gpus):
  jobs=[dict(condition=k,dim=dim,width=w,lr=lr,loss=loss) for k,dim,w,lr,loss in queue[j::len(gpus)]];path=O/('gpu'+str(gpu)+'_queue.json');dump(path,jobs);log=open(O/('gpu'+str(gpu)+'_training.log'),'a');p=subprocess.Popen([sys.executable,'-u',str(C/'neural_readouts.py'),'batch','--configfile',str(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT);active[gpu]=(p,log)
 while active:
  for gpu,(p,log) in list(active.items()):
   if p.poll() is not None:
    log.close()
    if p.returncode:raise RuntimeError('Readout GPU batch failed '+str(gpu))
    del active[gpu]
  completed=len(list(O.glob('*_d*/result.json')));dump(O/'state.json',{'stage':'full_training','completed':completed,'total':192,'active':{str(g):p.pid for g,(p,_) in active.items()}});time.sleep(5)
 report()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['manage','worker','batch']);ap.add_argument('--condition');ap.add_argument('--configfile');ap.add_argument('--dim',type=int);ap.add_argument('--width',type=int);ap.add_argument('--lr',type=float);ap.add_argument('--loss');a=ap.parse_args()
 try:
  if a.stage=='manage':manage()
  elif a.stage=='batch':
   for cfg in json.load(open(a.configfile)):worker(argparse.Namespace(**cfg))
  else:worker(a)
 except Exception as e:dump(O/('failure_'+str(a.condition)+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
