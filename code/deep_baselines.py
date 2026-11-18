"""Official TSlib deep numerical baselines, full budgets, validation-only selection."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import sys,json,time,subprocess,itertools,argparse,traceback,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from run import load,error,D,R,B,dump
C=B/'pycodes/AttentionPIT';OUT=R/'deep_baselines';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'pycodes/Time-Series-Library'))

def config(model,width):
 return SimpleNamespace(task_name='long_term_forecast',seq_len=28,label_len=14,pred_len=14,enc_in=1,dec_in=1,c_out=1,d_model=width,n_heads=8,e_layers=2 if width==128 else 3,d_layers=1,d_ff=4*width,dropout=.1,factor=3,activation='gelu',embed='timeF',freq='d',output_attention=False,down_sampling_window=2,down_sampling_layers=1,down_sampling_method='avg',channel_independence=1,moving_avg=7,decomp_method='moving_avg',top_k=5,use_norm=1)
def build(model,width):
 mod=__import__('models.'+model,fromlist=['Model']);cfg=config(model,width)
 return mod.Model(cfg,patch_len=7,stride=7) if model=='PatchTST' else mod.Model(cfg)
def data():
 rows,x,y=load();tr=np.array([i for i,z in enumerate(rows) if z['split']=='train']);va=np.array([i for i,z in enumerate(rows) if z['split']=='val']);mu=float(np.log1p(x[tr]).mean());std=max(float(np.log1p(x[tr]).std()),1e-5);xx=(np.log1p(x)-mu)/std;yy=(np.log1p(y)-mu)/std;lead=np.array([z['lead'] for z in rows]);return rows,x,y,tr,va,xx,yy,lead,mu,std

def work(a):
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 name=f'{a.model}_w{a.width}_lr{a.lr:g}_{a.loss}';dest=OUT/name;dest.mkdir(exist_ok=True)
 if (dest/'result.json').exists():return
 rows,x,y,tr,va,xx,yy,lead,mu,std=data();device='cuda';xt=torch.tensor(xx[:,:,None],device=device);yt=torch.tensor(yy,device=device);lt=torch.tensor(lead,device=device);m=build(a.model,a.width).to(device);opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);generator=np.random.default_rng(2024);best=np.inf;stale=0;curve=[];bestepoch=0
 def forward(ids):
  full=m(xt[ids],None,None,None)
  if isinstance(full,tuple):full=full[0]
  return full[:,:,0].gather(1,lt[ids,None]+torch.arange(7,device=device)[None,:])
 for epoch in range(1,151):
  m.train();losses=[]
  for ids in np.array_split(generator.permutation(tr),max(1,int(np.ceil(len(tr)/64)))):
   opt.zero_grad(set_to_none=True);pred=forward(ids);loss=torch.nn.functional.mse_loss(pred,yt[ids]) if a.loss=='logMSE' else torch.nn.functional.smooth_l1_loss(pred,yt[ids])
   if not torch.isfinite(loss):raise RuntimeError('Non-finite loss '+name)
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach()))
  m.eval()
  with torch.inference_mode():pv=torch.cat([forward(ids).cpu() for ids in np.array_split(va,max(1,int(np.ceil(len(va)/128))))]).numpy();pv=np.expm1(np.clip(pv*std+mu,0,25));score=float(error(y[va],pv).mean())
  curve.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_WAPE':score})
  if score<best-1e-8:best=score;bestepoch=epoch;stale=0;torch.save({'state_dict':m.state_dict(),'model':a.model,'width':a.width,'lr':a.lr,'loss':a.loss,'mean':mu,'std':std,'seed':2024},dest/'best.pt')
  else:stale+=1
  dump(dest/'curve.json',curve)
  if epoch%10==0:print(name,epoch,score,'BEST',bestepoch,best,flush=True)
  if epoch>=30 and stale>=20:break
 dump(dest/'result.json',{'model':a.model,'width':a.width,'lr':a.lr,'loss':a.loss,'seed':2024,'epochs_trained':epoch,'best_epoch':bestepoch,'validation_WAPE':best,'minimum_epochs':30,'maximum_epochs':150,'patience':20,'parameters':sum(p.numel() for p in m.parameters()),'official_source_commit':'79f4325dd69fc489a94ff53135800b136a8f49ae'});print('COMPLETE',name,flush=True)

def report():
 rows,x,y,tr,va,xx,yy,lead,mu,std=data();pred={};selection={};torch.set_num_threads(2)
 for model in ['PatchTST','iTransformer','TimeMixer']:
  paths=list(OUT.glob(model+'_*/result.json'));assert len(paths)==12,(model,len(paths));best=min(paths,key=lambda p:json.load(open(p))['validation_WAPE']);v=json.load(open(best));ck=torch.load(best.parent/'best.pt',map_location='cpu',weights_only=True);m=build(model,v['width']);m.load_state_dict(ck['state_dict']);m.eval();ps=[]
  with torch.inference_mode():
   for ids in np.array_split(np.arange(len(rows)),max(1,int(np.ceil(len(rows)/128)))):
    full=m(torch.tensor(xx[ids,:,None]),None,None,None);full=full[0] if isinstance(full,tuple) else full;pv=full[:,:,0].gather(1,torch.tensor(lead[ids,None])+torch.arange(7)[None,:]);ps.append(pv.numpy())
  pred[model]=np.maximum(np.expm1(np.clip(np.concatenate(ps)*std+mu,0,25)),0);selection[model]=v
 np.savez_compressed(OUT/'predictions.npz',**pred);fresh=set(json.load(open(R/'protocol.json')).get('fresh_test_pageids',[]));result={'seed':2024,'selected_by':'pooled validation WAPE; never testing unfinished configs','selected':selection,'curves':{}}
 for kind in ['event','calendar_control','fresh_event']:
  result['curves'][kind]={}
  for l in [7,3,1]:
   ii=[i for i,z in enumerate(rows) if z['split']=='test' and z['lead']==l and (z['kind']==kind or kind=='fresh_event' and z['kind']=='event' and z['pageid'] in fresh)];result['curves'][kind][str(l)]={k:{'WAPE':float(error(y[ii],p[ii]).mean()),'n':len(ii)} for k,p in pred.items()}
 dump(OUT/'results.json',result);print('ALL DEEP BASELINES COMPLETE',flush=True)

def manage():
 protocol={'models':['PatchTST','iTransformer','TimeMixer'],'lr':[.0001,.0003,.001],'width':[128,256],'loss':['logMSE','logHuber'],'min_epochs':30,'max_epochs':150,'patience':20,'batch':64,'seed':2024,'validation':'pooled macro WAPE, checkpoint and configuration; no test selection','input':'same28days, log1p with training-only global standardization','output':'14futuresteps; score only lead:lead+7 for each fixed target','official_source':'thuml/Time-Series-Library','commit':'79f4325dd69fc489a94ff53135800b136a8f49ae'};dump(OUT/'protocol.json',protocol);queue=list(itertools.product(protocol['models'],protocol['lr'],protocol['width'],protocol['loss']));active={};done=0
 while queue or active:
  for gpu,(p,key,log) in list(active.items()):
   if p.poll() is not None:
    log.close()
    if p.returncode:raise RuntimeError('Deep baseline failed '+str(key))
    done+=1;del active[gpu]
  for gpu in range(1,7):
   if gpu in active or not queue:continue
   model,lr,width,loss=queue.pop(0);key=f'{model}_w{width}_lr{lr:g}_{loss}'
   if (OUT/key/'result.json').exists():done+=1;continue
   log=open(OUT/(key+'.log'),'a');cmd=[sys.executable,'-u',str(C/'deep_baselines.py'),'worker','--model',model,'--lr',str(lr),'--width',str(width),'--loss',loss];p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT);active[gpu]=(p,key,log)
  dump(OUT/'state.json',{'completed':done,'queued':len(queue),'active':{str(g):{'pid':p.pid,'configuration':key} for g,(p,key,_) in active.items()}});time.sleep(5)
 report()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','manage']);p.add_argument('--model');p.add_argument('--lr',type=float);p.add_argument('--width',type=int);p.add_argument('--loss');a=p.parse_args()
 try:work(a) if a.stage=='worker' else manage()
 except Exception as e:dump(OUT/('failure_'+str(a.model)+'.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
