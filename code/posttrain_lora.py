"""Matched frozen vs task-supervised QLoRA; independent from SMCB."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import sys,json,time,argparse,subprocess,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from run import load,read_npz,snapshot_key,dump,D,R,B,error
C=B/'pycodes/AttentionPIT';O=R/'posttrain01';O.mkdir(exist_ok=True)
PATHS={'Qwen3':B/'models/ChronoGeom/qwen3-8b-b968826d','Mistral7B':B/'pycodes/WebBurst/weights/mistralai--Mistral-7B-Instruct-v0.2'}
def design():
 rows,x,y=load();d=read_npz(R/'neural_readouts/features.npz');f=d['Metadata_0'][:,:40];base=read_npz(R/'neural_readouts/all_predictions.npz')['NumericReadout'];target=np.log1p(y)-np.log1p(base)
 texts=[str(read_npz(D/'representations'/(snapshot_key(z)+'.npz'))['text']) for z in rows]
 return rows,f.astype('float32'),base,y,target.astype('float32'),texts,d['tr'],d['va']
def head(h):
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(2024)
  return torch.nn.ModuleDict({'project':torch.nn.Sequential(torch.nn.LayerNorm(h),torch.nn.Linear(h,32),torch.nn.GELU()),'predict':torch.nn.Sequential(torch.nn.Linear(72,64),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(64,64),torch.nn.GELU(),torch.nn.Linear(64,7))})
def loader(family,adapt):
 from transformers import AutoTokenizer,AutoModel,BitsAndBytesConfig
 from peft import prepare_model_for_kbit_training,get_peft_model,LoraConfig
 tok=AutoTokenizer.from_pretrained(PATHS[family],local_files_only=True);tok.padding_side='right'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 device_map={'':0}
 if os.environ.get('ATTENTION_PIT_SHARD2')=='1':
  config=json.load(open(PATHS[family]/'config.json'));layers=config['num_hidden_layers'];device_map={'embed_tokens':0,'rotary_emb':0,'norm':1,**{'layers.'+str(i):int(i>=layers//2) for i in range(layers)}}
 m=AutoModel.from_pretrained(PATHS[family],local_files_only=True,quantization_config=cfg,device_map=device_map,dtype=torch.bfloat16,attn_implementation='sdpa')
 m=prepare_model_for_kbit_training(m,use_gradient_checkpointing=adapt)
 if adapt:
  m=get_peft_model(m,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none'))
  m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 # Release only unused load/quantization workspace before training; model tensors retained.
 torch.cuda.empty_cache()
 return tok,m

def worker(a):
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024)
 name=f'{a.family}_{a.mode}_lr{a.lr:g}';dest=O/name;dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():return
 rows,f,base,y,target,texts,tr,va=design();tok,m=loader(a.family,a.mode=='lora');device='cuda';h=head(m.config.hidden_size).to(device);opt=torch.optim.AdamW([{'params':h.parameters(),'lr':1e-4}]+([{'params':[p for p in m.parameters() if p.requires_grad],'lr':a.lr}] if a.mode=='lora' else []),weight_decay=1e-4)
 tokens=[tok(t,return_tensors='pt',add_special_tokens=True) for t in texts];fx=torch.tensor(f,device=device);tt=torch.tensor(target,device=device)
 cachepath=O/(a.family+'_nf4_frozen_embeddings.npz')
 def encode(i):
  t={k:v.to(device) for k,v in tokens[i].items()};out=m(**t,use_cache=False).last_hidden_state;mask=t['attention_mask'].unsqueeze(-1).to(out.device);return ((out.float()*mask).sum(1)/mask.sum(1)).float().to(device)
 if a.mode=='frozen':
  m.eval()
  if cachepath.exists():emb=torch.tensor(read_npz(cachepath)['embedding'],device=device)
  else:
   out=[];unique_embeddings={}
   with torch.inference_mode():
    for i in range(len(rows)):
     digest=hashlib.sha256(texts[i].encode()).hexdigest()
     if digest not in unique_embeddings:unique_embeddings[digest]=encode(i).cpu().numpy()[0]
     out.append(unique_embeddings[digest])
     if i%50==0:print('FROZEN ENCODE',a.family,i,len(rows),flush=True)
   ar=np.stack(out);np.savez_compressed(cachepath,embedding=ar);emb=torch.tensor(ar,device=device)
 def predict(ids,textids=None):
  h.eval();m.eval();out=[];evaluation_embeddings={}
  with torch.inference_mode():
   for j,i in enumerate(ids):
    k=i if textids is None else textids[j]
    if a.mode=='frozen':e=emb[k:k+1]
    else:
     digest=hashlib.sha256(texts[k].encode()).hexdigest()
     if digest not in evaluation_embeddings:evaluation_embeddings[digest]=encode(k)
     e=evaluation_embeddings[digest]
    out.append(h['predict'](torch.cat([fx[i:i+1],h['project'](e)],-1)).cpu().numpy()[0])
  return np.maximum(np.expm1(np.clip(np.log1p(base[ids])+np.stack(out),0,25)),0)
 rng=np.random.default_rng(2024);best=float('inf');stale=0;curve=[];bestepoch=0
 dump(dest/'started.json',{'pid':os.getpid(),'visible_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),'two_gpu_model_sharding':os.environ.get('ATTENTION_PIT_SHARD2')=='1','family':a.family,'mode':a.mode,'lora_lr':a.lr,'train_rows':len(tr),'val_rows':len(va),'min_epochs':30,'max_epochs':150,'patience':20,'effective_batch':16,'max_input_tokens':max(t['input_ids'].numel() for t in tokens),'trainable_lm_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'head_parameters':sum(p.numel() for p in h.parameters())})
 for epoch in range(1,151):
  epoch_started=time.perf_counter();h.train();m.train(a.mode=='lora');losses=[];perm=rng.permutation(tr)
  for start in range(0,len(perm),16):
   ids=perm[start:start+16];opt.zero_grad(set_to_none=True)
   for i in ids:
    e=emb[i:i+1] if a.mode=='frozen' else encode(i);pred=h['predict'](torch.cat([fx[i:i+1],h['project'](e)],-1));loss=torch.nn.functional.smooth_l1_loss(pred,tt[i:i+1]);assert torch.isfinite(loss);(loss/len(ids)).backward();losses.append(float(loss.detach()))
   if epoch==1 and start==0:
    grads={n:float(p.grad.float().norm()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None};dump(dest/'gradient_audit.json',{'finite':all(np.isfinite(list(grads.values()))),'nonzero_lora_tensors':sum(v>0 for v in grads.values()),'lora_gradient_norms':grads});assert a.mode=='frozen' or any(v>0 for v in grads.values())
   torch.nn.utils.clip_grad_norm_(list(h.parameters())+[p for p in m.parameters() if p.requires_grad],1.);opt.step()
   if start%112==0:print('TRAIN_BATCH',name,'epoch',epoch,'rows_done',min(start+len(ids),len(perm)),'rows_total',len(perm),'elapsed_seconds',round(time.perf_counter()-epoch_started,2),flush=True)
  train_seconds=time.perf_counter()-epoch_started;validation_started=time.perf_counter();pv=predict(va);score=float(error(y[va],pv).mean());curve.append({'epoch':epoch,'train_logHuber':float(np.mean(losses)),'validation_WAPE':score,'train_seconds':train_seconds,'validation_seconds':time.perf_counter()-validation_started});dump(dest/'curve.json',curve)
  if score<best-1e-8:
   best=score;stale=0;bestepoch=epoch;torch.save(h.state_dict(),dest/'head.pt')
   if a.mode=='lora':
    m.save_pretrained(dest/'adapter');(dest/'adapter/README.md').unlink(missing_ok=True)
  else:stale+=1
  print('EPOCH',name,epoch,score,'BEST',best,bestepoch,flush=True)
  if epoch>=30 and stale>=20:break
 dump(dest/'result.json',{'family':a.family,'mode':a.mode,'adapter_lr':a.lr,'head_lr':1e-4,'seed':2024,'epochs_trained':epoch,'best_epoch':bestepoch,'validation_WAPE':best,'test_evaluated':False})

def manager():
 protocol={'seed':2024,'families':list(PATHS),'modes':['nf4_frozen_same_head','nf4_lora_r8_qkvo'],'adapter_lrs':[1e-5,5e-5],'head_lr':1e-4,'head':'LayerNorm-hidden-to32; concat numeric37+metadata3; MLP72-64-64-7','base':'frozen validation-selected NumericReadout strong baseline','inputs':'exact same retained historical texts as existing study; no additional truncation','objective':'smoothL1(log1p(y)-log1p(frozen_numeric_prediction))','training':{'min_epochs':30,'max_epochs':150,'patience':20,'batch':16,'microbatch':1},'selection':'pooled validation macroWAPE; no test used in selection','scope':'separate from SMCB; frozen and adapted each share family, nf4 quantization, pooling, predictor, input and training budget','evaluation':'after all six configurations complete; compare numeric baseline and previous frozen study, plus fixed-weight old/mismatched text diagnostics','test_status':'existing test previously inspected; exploratory comparison, not new untouched holdout'}
 dump(O/'protocol.json',protocol)
 queue=[('Qwen3','frozen',0.),('Qwen3','lora',1e-5),('Mistral7B','lora',1e-5),('Mistral7B','frozen',0.),('Qwen3','lora',5e-5),('Mistral7B','lora',5e-5)];active={};stable_free={}
 # Adopt only this exact worker command on restart.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   cmd=(proc/'cmdline').read_bytes().decode().split('\0');env=(proc/'environ').read_bytes().decode().split('\0')
   if str(C/'posttrain_lora.py') not in cmd or 'worker' not in cmd:continue
   f=cmd[cmd.index('--family')+1];mode=cmd[cmd.index('--mode')+1];lr=float(cmd[cmd.index('--lr')+1]);gpu=int(next(z.split('=',1)[1] for z in env if z.startswith('CUDA_VISIBLE_DEVICES=')));active[gpu]={'pid':int(proc.name),'job':(f,mode,lr),'adopted':True}
  except (OSError,ValueError,StopIteration,IndexError):pass
 while queue or active:
  pair=json.load(open(O/'pair_allocation.json')) if (O/'pair_allocation.json').exists() else {}
  delegated={tuple(v) for v in pair.get('configs',[])};queue=[v for v in queue if tuple(v) not in delegated]
  reserved_gpus=pair.get('gpus',[]) if not (O/'pair_complete.json').exists() else []
  if (O/'pair_failure.json').exists():raise RuntimeError('Two-GPU delegated worker failed; see pair_failure.json')
  for gpu,v in list(active.items()):
   try:
    os.waitpid(v['pid'],os.WNOHANG)
   except ChildProcessError:pass
   dead=not Path('/proc',str(v['pid'])).exists()
   if not dead:
    try:dead=(Path('/proc')/str(v['pid'])/'stat').read_text().split()[2]=='Z'
    except FileNotFoundError:dead=True
   if dead:
    f,mode,lr=v['job'];dest=O/f'{f}_{mode}_lr{lr:g}'
    if not (dest/'result.json').exists():raise RuntimeError('Worker failed '+str(v))
    del active[gpu]
  queue=[job for job in queue if not (O/f'{job[0]}_{job[1]}_lr{job[2]:g}'/'result.json').exists() and not any(tuple(v['job'])==job for v in active.values())]
  free={int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
  for gpu in [6,7,2,3,4,5]:
   stable_free[gpu]=stable_free.get(gpu,0)+1 if free[gpu]>=14000 and gpu not in active else 0
   if not queue:break
   if gpu in active or gpu in reserved_gpus or stable_free[gpu]<3:continue
   f,mode,lr=queue.pop(0);name=f'{f}_{mode}_lr{lr:g}';log=open(O/(name+'.log'),'a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');p=subprocess.Popen([sys.executable,'-u',str(C/'posttrain_lora.py'),'worker','--family',f,'--mode',mode,'--lr',str(lr)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close();active[gpu]={'pid':p.pid,'job':(f,mode,lr)}
  dump(O/'state.json',{'stage':'training_or_waiting_for_stable_14000MiB_free','queued':queue,'active':active,'complete_configs':len(list(O.glob('*/result.json'))),'total_configs':6});time.sleep(20)
 while len(list(O.glob('*/result.json')))<6:
  if (O/'pair_failure.json').exists():raise RuntimeError('Delegated pair failure')
  time.sleep(20)
 dump(O/'training_complete.json',{'complete':True,'configs':6,'next':'validation selection and paired inference diagnostics'})
 while True:
  free={int(z.split(',')[0]):int(z.split(',')[1]) for z in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
  g=next((g for g in [6,7,2,3,4,5] if free[g]>=14000),None)
  if g is not None:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g));subprocess.run([sys.executable,'-u',str(C/'posttrain_lora.py'),'report'],env=env,check=True)
   with open(R/'neighbors01/title_manager_after_posttrain.log','a') as log:
    resumed=subprocess.Popen([sys.executable,'-u',str(C/'neighbor_title_manage.py')],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
   dump(O/'resumed_title_manager.json',{'pid':resumed.pid,'after':'completed posttraining and paired test report'});break
  time.sleep(20)

def report():
 from peft import PeftModel
 torch.set_num_threads(2);rows,f,base,y,target,texts,tr,va=design();fx=torch.tensor(f,device='cuda');pred={'NumericReadout':base};old=read_npz(R/'neural_readouts/all_predictions.npz');pred.update({k:old[k] for k in ['Chronos2','PatchTST','iTransformer','TimeMixer','Qwen3','Mistral7B']});selected={};frozen=[next(j for j,w in enumerate(rows) if w['event_id']==z['event_id'] and w['kind']==z['kind'] and w['lead']==7) for z in rows]
 snaps=[json.load(open(D/'snapshots'/(snapshot_key(z)+'.json'))) for z in rows];donors=[]
 for i,z in enumerate(rows):
  pool=[j for j,w in enumerate(rows) if w['pageid']!=z['pageid'] and snaps[j]['available'] and snaps[j]['revision_time']<z['origin'] and w['origin']<=z['origin']]
  donors.append(min(pool,key=lambda j:(rows[j]['lead']!=z['lead'],abs(len(texts[j])-len(texts[i])),rows[j]['event_id'])) if pool else i)
 for family in PATHS:
  for mode in ['frozen','lora']:
   files=list(O.glob(f'{family}_{mode}_*/result.json'));assert len(files)==(1 if mode=='frozen' else 2)
   bestfile=min(files,key=lambda p:json.load(open(p))['validation_WAPE']);v=json.load(open(bestfile));name=family+'_'+mode;selected[name]=v
   cached=O/'completed_evaluation'/(bestfile.parent.name+'.npz');cached_meta=cached.with_suffix('.json')
   if cached.exists() and cached_meta.exists():
    cv=json.load(open(cached_meta));assert abs(cv['checkpoint_validation_WAPE']-v['validation_WAPE'])<.005
    cp=read_npz(cached)
    for suffix,key in [('', 'CurrentText'),('_FrozenWords7','OldText7d'),('_WrongWords','WrongText')]:
     assert cp[key].shape==base.shape;pred[name+suffix]=cp[key]
    continue
   tok,m=loader(family,False)
   if mode=='lora':m=PeftModel.from_pretrained(m,bestfile.parent/'adapter',is_trainable=False)
   m.eval();h=head(m.config.hidden_size).cuda();h.load_state_dict(torch.load(bestfile.parent/'head.pt',weights_only=True,map_location='cuda'));h.eval();cache={}
   def embedding(k):
    digest=hashlib.sha256(texts[k].encode()).hexdigest()
    if digest not in cache:
     t=tok(texts[k],return_tensors='pt').to('cuda');out=m(**t,use_cache=False).last_hidden_state;mask=t.attention_mask.unsqueeze(-1);cache[digest]=((out.float()*mask).sum(1)/mask.sum(1)).cpu()
    return cache[digest].cuda()
   with torch.inference_mode():
    for suffix,indices in [('',list(range(len(rows)))),('_FrozenWords7',frozen),('_WrongWords',donors)]:
     out=[]
     for i,k in enumerate(indices):out.append(h['predict'](torch.cat([fx[i:i+1],h['project'](embedding(k))],-1)).cpu().numpy()[0])
     pred[name+suffix]=np.maximum(np.expm1(np.clip(np.log1p(base)+np.stack(out),0,25)),0)
   del m,h;torch.cuda.empty_cache()
 np.savez_compressed(O/'predictions.npz',**pred);out={'seed':2024,'selection':selected,'curves':{},'gains':{},'scope':'single-seed exploratory; all pre-existing test results previously inspected'}
 fresh=set(json.load(open(R/'protocol.json'))['fresh_test_pageids'])
 for kind in ['event','calendar_control','fresh_event','changed_text_event']:
  out['curves'][kind]={};out['gains'][kind]={}
  for lead in [7,3,1]:
   ii=np.array([i for i,z in enumerate(rows) if z['split']=='test' and (z['kind']==kind or kind=='fresh_event' and z['kind']=='event' and z['pageid'] in fresh or kind=='changed_text_event' and z['kind']=='event' and texts[i]!=texts[frozen[i]]) and z['lead']==lead]);
   if not len(ii):continue
   groups=np.array([rows[i]['pageid'] for i in ii]);unique=np.unique(groups);rng=np.random.default_rng(2024);boots=[np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)]) for _ in range(2000)];es={k:error(y[ii],p[ii]) for k,p in pred.items()};out['curves'][kind][str(lead)]={k:{'WAPE':float(e.mean()),'RMSLE':float(np.sqrt(np.mean((np.log1p(y[ii])-np.log1p(pred[k][ii]))**2))),'n':len(ii)} for k,e in es.items()};gg={}
   for family in PATHS:
    for a,b in [('NumericReadout',family+'_lora'),(family+'_frozen',family+'_lora'),(family+'_lora_FrozenWords7',family+'_lora'),(family+'_lora_WrongWords',family+'_lora')]:
     delta=es[a]-es[b];gg[a+' -> '+b]={'pp':float(delta.mean()),'CI95':np.quantile([delta[j].mean() for j in boots],[.025,.975]).tolist()}
   out['gains'][kind][str(lead)]=gg
 dump(O/'results.json',out)
 import csv
 with open(O/'comparison.csv','w',newline='') as file:
  writer=csv.writer(file);writer.writerow(['cohort','lead_days','model','n','WAPE_percent','RMSLE'])
  for kind,leads in out['curves'].items():
   for lead,models in leads.items():
    for model,metrics in models.items():writer.writerow([kind,lead,model,metrics['n'],metrics['WAPE'],metrics['RMSLE']])
 with open(O/'semantic_gains.csv','w',newline='') as file:
  writer=csv.writer(file);writer.writerow(['cohort','lead_days','comparison','WAPE_gain_pp','CI95_low','CI95_high'])
  for kind,leads in out['gains'].items():
   for lead,comparisons in leads.items():
    for comparison,metrics in comparisons.items():writer.writerow([kind,lead,comparison,metrics['pp'],*metrics['CI95']])
 print('POSTTRAIN COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['manager','worker','report']);p.add_argument('--family');p.add_argument('--mode');p.add_argument('--lr',type=float,default=0);a=p.parse_args()
 try:
  if a.stage=='manager':manager()
  elif a.stage=='report':report()
  else:worker(a)
 except Exception as e:dump(O/('failure_'+str(a.family)+'_'+str(a.mode)+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
