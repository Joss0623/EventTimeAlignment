"""SMCB temporal adaptation. Source: Zhao et al., SGFNet, arXiv2609.27854,
DOI10.1109/LGRS.2026.3726463, official commit987f46a4a3087ab4477433068a0b8a7095b9e8f9.
Changes: 1D temporal windows; unordered historical-link context; optional language
conditioning. Original SMCB is credited, not claimed as an original contribution.
"""
import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,time,copy,itertools,argparse,traceback,math
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from run import load,error,R,D,read_npz,dump
BASE=R/'neural_readouts';O=R/'neighbors01';G=D/'neighbors01';O.mkdir(exist_ok=True)
KEYS=['SMCB_Target','SMCB_TargetQwen3','Static_Neighbors','SMCB_Neighbors','SMCB_Metadata','SMCB_Qwen3ReadoutOnly','SMCB_Qwen3','SMCB_Mistral7B','SMCB_GPToss20B','SMCB_DeepSeekV2Lite','SMCB_KimiVLA3B']
FAMILY={'SMCB_TargetQwen3':'Qwen3','SMCB_Qwen3ReadoutOnly':'Qwen3',**{'SMCB_'+k:k for k in ['Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B']}}
class TemporalSMCB(nn.Module):
 def __init__(self,c,k,static=False):
  super().__init__();self.c=c;self.k=k;self.static=static;self.q=nn.Conv1d(c,c,1,bias=False);self.key=nn.Conv1d(c,c,1,bias=False);self.value=nn.Conv1d(c,c,1,bias=False);self.kernel_proj=nn.Linear(k,k);self.depthwise=nn.Conv1d(c,c,3,padding=1,groups=c,bias=False);self.language_query=nn.Linear(35,c,bias=False);self.norm=nn.LayerNorm(c)
 def forward(self,x,mask,text,title):
  b,n,c,t=x.shape;flat=x.reshape(b*n,c,t);q=self.q(flat).reshape(b,n,c,t)+self.language_query(text)[:,None,:,None];key=self.key(F.adaptive_avg_pool1d(flat,self.k)).reshape(b,n,c,self.k)+self.language_key(title)[:,:,:,None]
  # Reference-node log-mean-exp is permutation invariant. Kernel bins represent time, never page ordering.
  affinity=torch.einsum('bnct,bmcp->bntmp',q,key)/math.sqrt(c);affinity=affinity.masked_fill(~mask[:,None,None,:,None],-1e4);affinity=torch.logsumexp(affinity,dim=3)-mask.sum(1).clamp_min(1).log()[:,None,None,None];weights=self.kernel_proj(affinity).softmax(-1)
  if self.static:weights=torch.ones_like(weights)/self.k
  values=self.value(flat);patch=F.unfold(values.unsqueeze(2),kernel_size=(1,self.k),padding=(0,self.k//2)).reshape(b,n,c,self.k,t).permute(0,1,2,4,3);dynamic=torch.einsum('bntk,bnctk->bnct',weights,patch);local=self.depthwise(flat).reshape(b,n,c,t);out=F.gelu(self.norm((dynamic+local).permute(0,1,3,2))).permute(0,1,3,2);return out*mask[:,:,None,None],weights
class Forecaster(nn.Module):
 def __init__(self,width,k,static=False,semantic=True):
  super().__init__();c=width//4;self.semantic=semantic;self.embed=nn.Linear(4,c);self.blocks=nn.ModuleList([TemporalSMCB(c,k,static) for _ in range(2)]);self.read_query=nn.Linear(c+35,c);self.read_key=nn.Linear(c,c,bias=False);self.project=nn.Linear(c*28,32);self.head=nn.Sequential(nn.Linear(104,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,width),nn.GELU(),nn.Linear(width,7));self.c=c
  # Append semantic-identity paths after the original network initialization.
  # Preserve its RNG state and exact own-page-only function (zero neighbor-title inputs).
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(2024)
   for block in self.blocks:block.language_key=nn.Linear(32,c,bias=False)
   self.read_key_title=nn.Linear(32,c,bias=False);self.title_head=nn.Linear(32,32,bias=False)
 def forward(self,nodes,mask,f,titles=None):
  # nodes[B,N,T,4]; all N histories stop strictly before the same forecast origin.
  titles=torch.zeros(nodes.shape[0],nodes.shape[1],32,device=nodes.device) if titles is None else titles;semantic_titles=titles if self.semantic else torch.zeros_like(titles)
  x=F.gelu(self.embed(nodes)).permute(0,1,3,2)*mask[:,:,None,None];text=f[:,-35:] if self.semantic else torch.zeros_like(f[:,-35:])
  for block in self.blocks:x,_=block(x,mask,text,semantic_titles)
  pooled=x.mean(-1);q=self.read_query(torch.cat([pooled[:,0],text],1));scores=torch.einsum('bc,bnc->bn',q,self.read_key(pooled)+self.read_key_title(semantic_titles))/math.sqrt(self.c);weights=scores.masked_fill(~mask,-1e4).softmax(-1);context=torch.einsum('bn,bnct->bct',weights,x);nm=mask.clone();nm[:,0]=False;summary=(titles*nm[:,:,None]).sum(1)/nm.sum(1).clamp_min(1)[:,None];z=self.project(context.flatten(1))+self.title_head(summary);return self.head(torch.cat([f,z],1))
def inputs(key):
 rows,x,y=load();d=read_npz(BASE/'features.npz');sel=json.load(open(BASE/'selected_configs.json'));family=FAMILY.get(key)
 if family:f=d[family+'_'+str(sel[family]['dim'])].copy()
 elif key=='SMCB_Metadata':f=d['Metadata_0'].copy()
 else:f=d['NumericReadout_0'].copy()
 if key.startswith('SMCB_Target'):
  counts=np.zeros((len(x),5,28),np.float32);counts[:,0]=x;mask=np.zeros((len(x),5),bool);mask[:,0]=True;wrong=None
 else:
  ar=read_npz(G/'histories.npz');counts=ar['counts'];mask=ar['mask'];wrong=(ar['wrong_counts'],ar['wrong_mask'])
 titles=np.zeros((len(rows),5,32),np.float32)
 if family and not key.startswith('SMCB_Target'):
  source=np.load(G/(family+'_title_features.npy'));ix=read_npz(G/'histories.npz')['title_index'];titles=source[np.maximum(ix,0)];titles*=((ix>=0)&mask)[:,:,None];titles[:,0]=0
 return rows,d,f,counts,mask,wrong,titles

def nodefeatures(counts,mask):
 med=np.maximum(np.median(counts,axis=-1,keepdims=True),1);z=np.zeros((*counts.shape,4),np.float32);z[:,:,:,0]=(np.log1p(counts)-np.log1p(med))/3;z[:,:,:,1]=np.log1p(med)/10;z[:,:,:,2]=mask[:,:,None];z[:,0,:,3]=1;return z*mask[:,:,None,None]

def trial_name(key,w,k,lr,loss):return f'{key}_w{w}_k{k}_lr{lr:g}_{loss}'
def work(key,w,k,lr,loss):
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 dest=O/'trials'/trial_name(key,w,k,lr,loss);dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return
 rows,d,f,counts,mask,_,titles=inputs(key);tr=d['tr'];va=d['va'];nodes=torch.tensor(nodefeatures(counts,mask),device='cuda');ms=torch.tensor(mask,device='cuda');ft=torch.tensor(f,device='cuda');ts=torch.tensor(titles,device='cuda');target=torch.tensor(d['target'],device='cuda');model=Forecaster(w,k,static=key=='Static_Neighbors',semantic=key!='SMCB_Qwen3ReadoutOnly').cuda();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);rng=np.random.default_rng(2024);best=np.inf;stale=0;curve=[]
 for epoch in range(1,151):
  model.train();ls=[]
  for ids in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/64)))):
   opt.zero_grad(set_to_none=True);p=model(nodes[ids],ms[ids],ft[ids],ts[ids]);z=F.mse_loss(p,target[ids]) if loss=='logMSE' else F.smooth_l1_loss(p,target[ids]);assert torch.isfinite(z);z.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ls.append(float(z.detach()))
  model.eval()
  with torch.inference_mode():pv=np.concatenate([model(nodes[ids],ms[ids],ft[ids],ts[ids]).cpu().numpy() for ids in np.array_split(va,max(1,int(np.ceil(len(va)/128))))]);pred=np.expm1(np.clip(np.log1p(d['base'][va])+pv,0,25));score=float(error(d['y'][va],pred).mean())
  if score<best-1e-8:best=score;be=epoch;stale=0;torch.save(model.state_dict(),dest/'best.pt')
  else:stale+=1
  curve.append({'epoch':epoch,'loss':float(np.mean(ls)),'validation_WAPE':score});dump(dest/'curve.json',curve)
  if epoch%10==0:print(dest.name,epoch,score,flush=True)
  if epoch>=30 and stale>=20:break
 dump(dest/'result.json',{'condition':key,'width':w,'kernel':k,'lr':lr,'loss':loss,'seed':2024,'epochs_trained':epoch,'best_epoch':be,'validation_WAPE':best,'parameters':sum(p.numel() for p in model.parameters()),'depth':2,'source_SMCB':'SGFNet2609.27854','adaptation':'temporal1D + unordered historical linked-page references + language-conditioned kernel queries'})
 print('COMPLETE',dest.name,flush=True)

def report():
 torch.set_num_threads(2);pred={};selected={}
 for key in KEYS:
  files=list((O/'trials').glob(key+'_w*/result.json'));assert len(files)==24,(key,len(files));path=min(files,key=lambda p:json.load(open(p))['validation_WAPE']);v=json.load(open(path));selected[key]=v;rows,d,f,counts,mask,wrong,titles=inputs(key);model=Forecaster(v['width'],v['kernel'],static=key=='Static_Neighbors',semantic=key!='SMCB_Qwen3ReadoutOnly');state=torch.load(path.parent/'best.pt',map_location='cpu',weights_only=True);report=model.load_state_dict(state,strict=False);assert not report.unexpected_keys;assert all('language_key' in z or 'read_key_title' in z or 'title_head' in z for z in report.missing_keys);model.eval()
  def infer(cc,mm,ff,ss=None):
   ss=titles if ss is None else ss
   nodes=nodefeatures(cc,mm)
   with torch.inference_mode():pv=np.concatenate([model(torch.tensor(nodes[ids]),torch.tensor(mm[ids]),torch.tensor(ff[ids]),torch.tensor(ss[ids])).numpy() for ids in np.array_split(np.arange(len(rows)),max(1,int(np.ceil(len(rows)/128))))])
   return np.maximum(np.expm1(np.clip(np.log1p(d['base'])+pv,0,25)),0)
  pred[key]=infer(counts,mask,f)
  if wrong is not None:
   wrong_titles=np.zeros_like(titles)
   if key in FAMILY:
    ix=read_npz(G/'histories.npz')['wrong_title_index'];lookup=np.load(G/(FAMILY[key]+'_title_features.npy'));wrong_titles=lookup[np.maximum(ix,0)];wrong_titles*=((ix>=0)&wrong[1])[:,:,None];wrong_titles[:,0]=0
   pred[key+'WrongNeighbors']=infer(*wrong,f,wrong_titles)
   frozen=np.array([next(j for j,z in enumerate(rows) if z['event_id']==r['event_id'] and z['kind']==r['kind'] and z['lead']==7) for r in rows]);cc=counts.copy();mm=mask.copy();cc[:,1:]=counts[frozen,1:];mm[:,1:]=mask[frozen,1:];pred[key+'FrozenNeighbors7d']=infer(cc,mm,f)
  if key in FAMILY:
   family=FAMILY[key];dim=json.load(open(BASE/'selected_configs.json'))[family]['dim'];ff=read_npz(BASE/'features.npz')[family+'FixedMismatch_'+str(dim)];pred[key+'WrongWords']=infer(counts,mask,ff)
   frozen=np.array([next(j for j,z in enumerate(rows) if z['event_id']==r['event_id'] and z['kind']==r['kind'] and z['lead']==7) for r in rows]);ff=f.copy();ff[:,-32:]=f[frozen,-32:];pred[key+'FrozenWords7d']=infer(counts,mask,ff)
  print('REPORT',key,flush=True)
 old=read_npz(BASE/'all_predictions.npz');pred.update({k:old[k] for k in ['Chronos2','PatchTST','iTransformer','TimeMixer','NumericReadout','Metadata','TFIDF','Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B']});np.savez_compressed(O/'all_predictions.npz',**pred);fresh=set(json.load(open(R/'protocol.json'))['fresh_test_pageids']);out={'seed':2024,'selection':selected,'curves':{},'gains':{},'protocol':json.load(open(O/'protocol.json'))}
 for kind in ['event','fresh_event','calendar_control']:
  out['curves'][kind]={};out['gains'][kind]={}
  for lead in [7,3,1]:
   ii=np.array([i for i,z in enumerate(rows) if z['split']=='test' and z['lead']==lead and z['kind']==('event' if kind=='fresh_event' else kind) and (kind!='fresh_event' or z['pageid'] in fresh)]);groups=np.array([rows[i]['pageid'] for i in ii]);unique=np.unique(groups);rng=np.random.default_rng(2024);boots=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(2000)];es={k:error(d['y'][ii],p[ii]) for k,p in pred.items()};out['curves'][kind][str(lead)]={k:{'WAPE':float(z.mean()),'n':len(ii)} for k,z in es.items()};out['gains'][kind][str(lead)]={}
   for key in KEYS:
    refs=['NumericReadout','SMCB_Target','Static_Neighbors','SMCB_Neighbors','SMCB_Metadata']+[key+s for s in ['WrongNeighbors','FrozenNeighbors7d','WrongWords','FrozenWords7d'] if key+s in es]
    for ref in refs:
     delta=es[ref]-es[key];out['gains'][kind][str(lead)][ref+' -> '+key]={'pp':float(delta.mean()),'CI95':np.quantile([delta[b].mean() for b in boots],[.025,.975]).tolist()}
 dump(O/'results.json',out);print('FULL SMCB RESULTS COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--key');ap.add_argument('--width',type=int);ap.add_argument('--kernel',type=int);ap.add_argument('--lr',type=float);ap.add_argument('--loss');a=ap.parse_args()
 try:
  if a.stage=='worker':work(a.key,a.width,a.kernel,a.lr,a.loss)
  elif a.stage=='report':report()
 except Exception as e:dump(O/('failure_'+str(a.key)+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
