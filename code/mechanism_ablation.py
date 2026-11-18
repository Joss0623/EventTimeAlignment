"""Controlled residual-head ablations: no LLM retraining or new input information."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import sys,argparse,json,time,traceback,hashlib
from pathlib import Path
import numpy as np,torch
from posttrain_lora import design,head,O,R,D,error,dump,read_npz,snapshot_key
A=O/'mechanism_ablation01';A.mkdir(exist_ok=True)
CONDITIONS=['NumericOnly','MetadataOnly','CurrentText','OldText7d','WrongText','ZeroInitCurrent']
EXTENSION=['CurrentText128','MetadataOnly128','DeltaText','WrongDelta']
def head_for(hidden,dim=32):
 if dim==32:return head(hidden)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(2024)
  return torch.nn.ModuleDict({'project':torch.nn.Sequential(torch.nn.LayerNorm(hidden),torch.nn.Linear(hidden,dim),torch.nn.GELU()),'predict':torch.nn.Sequential(torch.nn.Linear(40+dim,64),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(64,64),torch.nn.GELU(),torch.nn.Linear(64,7))})
def delta_controls(raw,old,rows,tr,snap,texts):
 delta=raw-raw[old];norm=np.linalg.norm(delta,axis=1);wrong=np.zeros_like(delta);fallback=0
 for i,z in enumerate(rows):
  if norm[i]<=1e-8:continue
  pool=[j for j in tr if rows[j]['pageid']!=z['pageid'] and snap[j]['available'] and snap[j]['revision_time']<z['origin'] and rows[j]['origin']<=z['origin'] and norm[j]>1e-8]
  if not pool:fallback+=1;continue
  j=min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(float(norm[j])-float(norm[i])),abs(len(texts[j])-len(texts[i])),rows[j]['event_id']));wrong[i]=delta[j]*(norm[i]/norm[j])
 return delta,wrong,fallback

def worker(family,extension=False,init128=False):
 global A
 if init128:A=A/'initialization_repair';A.mkdir(exist_ok=True)
 torch.set_num_threads(2);rows,f,base,y,target,texts,tr,va=design();raw=read_npz(O/(family+'_nf4_frozen_embeddings.npz'))['embedding'];snap=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];old=np.array([next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]);donors=[]
 for i,z in enumerate(rows):
  pool=[j for j in tr if rows[j]['pageid']!=z['pageid'] and snap[j]['available'] and snap[j]['revision_time']<z['origin'] and rows[j]['origin']<=z['origin']]
  donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else -1)
 dnp=np.array([raw[j] if j>=0 else np.zeros(raw.shape[1],np.float32) for j in donors],np.float32);tt=torch.tensor(target,device='cuda');embs={k:torch.tensor(v,device='cuda') for k,v in [('CurrentText',raw),('OldText7d',raw[old]),('WrongText',dnp)]};embs['ZeroInitCurrent']=embs['CurrentText'];embs['NumericOnly']=torch.zeros_like(embs['CurrentText']);embs['MetadataOnly']=embs['NumericOnly'];fx={k:torch.tensor(f.copy(),device='cuda') for k in CONDITIONS+EXTENSION+['ZeroInitCurrent128','ZeroInitMetadata128']};fx['NumericOnly'][:,37:]=0
 if extension or init128:
  delta,wrongdelta,fallback=delta_controls(raw,old,rows,tr,snap,texts);embs.update({'CurrentText128':embs['CurrentText'],'MetadataOnly128':embs['MetadataOnly'],'DeltaText':torch.tensor(delta,device='cuda'),'WrongDelta':torch.tensor(wrongdelta,device='cuda')});dump(A/(family+'_extension_coverage.json'),{'nonzero_delta_rows':int((np.linalg.norm(delta,axis=1)>1e-8).sum()),'zero_fallback_nonzero_receiver':int(fallback),'wrong_delta':'asof training donor; same nonzero status, receiver norm preserved'})
 dump(A/(family+'_coverage.json'),{'wrong_text_donor_pool':'training pages only and already available asof recipient','zero_fallback':{s:int(sum(donors[i]<0 for i,z in enumerate(rows) if z['split']==s)) for s in ['train','val','test']},'rows':len(rows)})
 if init128:embs.update({'ZeroInitCurrent128':embs['CurrentText'],'ZeroInitMetadata128':embs['MetadataOnly']})
 for condition in (['ZeroInitCurrent128','ZeroInitMetadata128'] if init128 else EXTENSION if extension else CONDITIONS):
  for lr in [1e-5,3e-5,1e-4]:
   dest=A/f'{family}_{condition}_lr{lr:g}';dest.mkdir(exist_ok=True)
   if (dest/'result.json').exists():continue
   if condition=='CurrentText' and lr==1e-4:
    source=O/f'{family}_frozen_lr0';v=json.load(open(source/'result.json'));dump(dest/'result.json',{**v,'ablation_condition':condition,'lr':lr,'checkpoint_source':str(source/'head.pt'),'reused_existing_complete_config':True});continue
   torch.manual_seed(2024);np.random.seed(2024);h=head_for(raw.shape[1],128 if condition.endswith('128') else 32).cuda()
   if condition.startswith('ZeroInit'):torch.nn.init.zeros_(h['predict'][-1].weight);torch.nn.init.zeros_(h['predict'][-1].bias)
   opt=torch.optim.AdamW(h.parameters(),lr=lr,weight_decay=1e-4);rng=np.random.default_rng(2024);best=float('inf');stale=0;curve=[];be=0
   for epoch in range(1,151):
    h.train();perm=rng.permutation(tr);losses=[]
    for start in range(0,len(perm),16):
     ids=perm[start:start+16];opt.zero_grad(set_to_none=True)
     for i in ids:
      p=h['predict'](torch.cat([fx[condition][i:i+1],h['project'](embs[condition][i:i+1])],-1));loss=torch.nn.functional.smooth_l1_loss(p,tt[i:i+1]);assert torch.isfinite(loss);(loss/len(ids)).backward();losses.append(float(loss.detach()))
     torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
    h.eval()
    with torch.inference_mode():
     v=h['predict'](torch.cat([fx[condition][va],h['project'](embs[condition][va])],-1)).cpu().numpy();pv=np.maximum(np.expm1(np.clip(np.log1p(base[va])+v,0,25)),0);score=float(error(y[va],pv).mean())
    curve.append({'epoch':epoch,'train_logHuber':float(np.mean(losses)),'validation_WAPE':score})
    if score<best-1e-8:best=score;stale=0;be=epoch;torch.save(h.state_dict(),dest/'head.pt')
    else:stale+=1
    dump(dest/'curve.json',curve)
    if epoch>=30 and stale>=20:break
   dump(dest/'result.json',{'family':family,'ablation_condition':condition,'projection_dim':128 if condition.endswith('128') else 32,'lr':lr,'seed':2024,'epochs_trained':epoch,'best_epoch':be,'validation_WAPE':best,'checkpoint_source':str(dest/'head.pt'),'reused_existing_complete_config':False});print('ABLATION COMPLETE',family,condition,lr,epoch,best,flush=True)
 dump(A/(family+('_init128_complete.json' if init128 else '_extension_complete.json' if extension else '_complete.json')),{'complete':True,'configs':6 if init128 else 12 if extension else 18,'seed':2024})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--extension',action='store_true');p.add_argument('--init128',action='store_true');a=p.parse_args()
 try:worker(a.family,a.extension,a.init128)
 except Exception as e:dump(A/(a.family+('_init128_failure.json' if a.init128 else '_extension_failure.json' if a.extension else '_failure.json')),{'error':repr(e),'traceback':traceback.format_exc()});raise
