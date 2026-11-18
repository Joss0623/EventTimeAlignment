import os,json,hashlib,sys,time,subprocess,traceback
from pathlib import Path
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';D=B/'data/AttentionPIT/expanded01';R=B/'runs/AttentionPIT/expanded01';OLD=B/'data/AttentionPIT/pilot01';env=dict(os.environ,ATTENTION_PIT_DATA_ROOT=str(D),ATTENTION_PIT_RUN_ROOT=str(R),PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');PY=B/'venvs/attentionpit/bin/python'
def dump(name,value):
 p=R/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
def start(name,args,py=PY,gpus=''):
 log=open(R/(name+'.log'),'a');p=subprocess.Popen([str(py),'-u',*map(str,args)],cwd=C,env=dict(env,CUDA_VISIBLE_DEVICES=gpus),stdout=log,stderr=subprocess.STDOUT);print('START',name,p.pid,gpus,flush=True);return p
jobs={};stage='starting'
try:
 # Reuse earlier embeddings of byte-identical input strings, not additional model downloads.
 import numpy as np
 old=json.load(open(OLD/'cohort.json'));rr=[z for z in old['rows'] if z['numeric_complete']];texts=[]
 for z in rr:
  key=hashlib.sha256((str(z['pageid'])+'|'+z['origin']).encode()).hexdigest()[:24];texts.append(str(np.load(OLD/'representations'/(key+'.npz'))['text']))
 for key,fn in [('mistral','mistral'),('gpt-oss-20b','gpt-oss-20b'),('deepseek-v2-lite-chat','deepseek-v2-lite-chat'),('kimi-vl-a3b-instruct','kimi-vl-a3b-instruct')]:
  a=np.load(OLD/(fn+'_embeddings.npz'))['embedding'];assert len(a)==len(texts);cd=D/(key+'_representations');cd.mkdir(exist_ok=True)
  for text,vec in zip(texts,a):
   f=cd/(hashlib.sha256(text.encode()).hexdigest()+'.npy')
   if not f.exists():np.save(f,vec)
 if not (R/'numeric_predictions.npz').exists():jobs['numerical']=start('numerical',[C/'run.py','numerical'],B/'venvs/webburst-doublecount01/bin/python','1')
 if not (R/'embedding_source.json').exists():jobs['qwen']=start('qwen',[C/'run.py','embed'],gpus='0')
 stage='snapshots_and_primary_models'
 while not (D/'ready.json').exists() or any(p.poll() is None for p in jobs.values()):
  for name,p in jobs.items():
   if p.poll() not in [None,0]:raise RuntimeError(name+' failed; see '+name+'.log')
  if (R/'collection_failure.json').exists():raise RuntimeError('Snapshot collection failed; see collection_failure.json')
  dump('state.json',{'stage':stage,'pids':{k:p.pid for k,p in jobs.items()},'snapshots':len(list((D/'snapshots').glob('*.json'))),'representations':len(list((D/'representations').glob('*.npz')))});time.sleep(20)
 for name,p in jobs.items():
  if p.wait()!=0:raise RuntimeError(name+' failed')
 queue=[('gpt-oss-20b',3,'gpt-oss-20b_evaluate.py',PY),('deepseek-v2-lite-chat',2,'deepseek-v2-lite-chat_evaluate.py',PY),('kimi-vl-a3b-instruct',2,'kimi-vl-a3b-instruct_evaluate.py',B/'venvs/attentionpit-kimi/bin/python'),('mistral',1,'mistral.py',PY)];active={};completed=[];stage='parallel_frozen_encoders'
 while queue or active:
  for k,p in list(active.items()):
   if p.poll() is not None:
    if p.returncode:raise RuntimeError(k+' encoder failed')
    completed.append(k);del active[k];print('COMPLETE',k,flush=True)
  free=[]
  for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines():
   i,m=map(int,line.split(','))
   if m>=21000:free.append(i)
  for job in list(queue):
   k,n,fn,py=job
   if (D/(k+'_embeddings.npz')).exists():completed.append(k);queue.remove(job);continue
   if len(free)<n:continue
   gp=','.join(map(str,free[:n]));free=free[n:];args=[C/fn]+([k] if k!='mistral' else [])+['--embed-only'];active[k]=start(k,args,py,gp);queue.remove(job)
  dump('state.json',{'stage':stage,'completed':completed,'pending':[z[0] for z in queue],'pids':{k:v.pid for k,v in active.items()}});time.sleep(20)
 stage='readouts_and_full_validation_search';dump('state.json',{'stage':stage})
 for name,args,py in [('analyze',[C/'run.py','analyze'],PY),('mistral_analysis',[C/'mistral.py'],PY),('gpt_analysis',[C/'gpt-oss-20b_evaluate.py','gpt-oss-20b'],PY),('deepseek_analysis',[C/'deepseek-v2-lite-chat_evaluate.py','deepseek-v2-lite-chat'],PY),('kimi_analysis',[C/'kimi-vl-a3b-instruct_evaluate.py','kimi-vl-a3b-instruct'],B/'venvs/attentionpit-kimi/bin/python'),('tuning',[C/'tune.py'],PY)]:
  p=start(name,args,py)
  if p.wait():raise RuntimeError(name+' failed')
 dump('state.json',{'stage':'complete','seed':2024,'results':str(R/'tuning01/results.json')});print('ALL EXPERIMENTS COMPLETE',flush=True)
except Exception as e:dump('failure.json',{'stage':stage,'error':repr(e),'traceback':traceback.format_exc()});raise
