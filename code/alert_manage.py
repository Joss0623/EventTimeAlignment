import os,time,subprocess,json,itertools,traceback
from pathlib import Path
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';D=B/'data/AttentionPIT/alert01';R=B/'runs/AttentionPIT/alert01';PY=B/'venvs/attentionpit/bin/python'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
def dump(x):
 p=R/'training_state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
keys=['PatchTST','iTransformer','TimeMixer','Numeric','Metadata','TFIDF','Qwen3','Mistral','gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct','Chronos2'];queue=[(key,w,lr,pw) for key in keys for w,lr,pw in itertools.product([128,256] if key in keys[:3] else [64,128],[.0001,.0003,.001],[0,1])];active={};done=0
class Adopted:
 def __init__(self,pid,name):self.pid=pid;self.name=name;self.returncode=None
 def poll(self):
  try:
   state=Path('/proc')/str(self.pid)/'stat'
   if state.read_text().split()[2]!='Z':return None
  except (FileNotFoundError,ProcessLookupError):pass
  self.returncode=0 if (R/'trials'/self.name/'result.json').exists() else 1
  return self.returncode
class NoLog:
 def close(self):pass
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
  if not any(Path(v).name=='alert_train.py' for v in args) or 'worker' not in args:continue
  k=args[args.index('--key')+1];w=int(args[args.index('--width')+1]);lr=float(args[args.index('--lr')+1]);pw=int(args[args.index('--pw')+1]);task=(k,w,lr,pw)
  envproc=dict(v.split('=',1) for v in (proc/'environ').read_bytes().decode().split('\0') if '=' in v);gpu=int(envproc['CUDA_VISIBLE_DEVICES']);name=f'{k}_w{w}_lr{lr:g}_pw{pw}'
  if task in queue:queue.remove(task)
  active[gpu]=(Adopted(int(proc.name),name),task,NoLog());print('ADOPT',proc.name,task,gpu,flush=True)
 except (FileNotFoundError,PermissionError,ValueError,KeyError):continue
for task in list(queue):
 k,w,lr,pw=task
 if (R/'trials'/f'{k}_w{w}_lr{lr:g}_pw{pw}'/'result.json').exists():queue.remove(task);done+=1
try:
 while queue or active:
  for gpu,(p,task,log) in list(active.items()):
   if p.poll() is not None:
    log.close()
    if p.returncode:raise RuntimeError(str(task)+' failed exit '+str(p.returncode))
    done+=1;del active[gpu]
  free={int(z.split(',')[0]):int(z.split(',')[1]) for z in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
  for gpu in [2,3,4,5]:
   if gpu in active or free[gpu]<3200:continue
   pick=None
   for task in queue:
    k,w,lr,pw=task
    if k not in keys[:5] and not (D/(k+'_features.npy')).exists():continue
    if k in keys[:3] and free[gpu]<3200:continue
    pick=task;break
   if pick is None:continue
   queue.remove(pick);k,w,lr,pw=pick;name=f'{k}_w{w}_lr{lr:g}_pw{pw}'
   if (R/'trials'/name/'result.json').exists():done+=1;continue
   log=open(R/(name+'.log'),'a');p=subprocess.Popen([str(PY),'-u',str(C/'alert_train.py'),'worker','--key',k,'--width',str(w),'--lr',str(lr),'--pw',str(pw)],cwd=C,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT);active[gpu]=(p,pick,log)
  dump({'complete':False,'completed':done,'queued':len(queue),'active':{str(g):{'pid':p.pid,'task':task} for g,(p,task,_) in active.items()},'budget':{'minimum_epochs':30,'maximum_epochs':150,'patience':20},'seed':2024})
  time.sleep(10)
 with open(R/'final_analysis.log','a') as log:
  for stage in ['controls','report']:
   p=subprocess.run([str(PY),'-u',str(C/'alert_train.py'),stage],cwd=C,env=env,stdout=log,stderr=subprocess.STDOUT)
   if p.returncode:raise RuntimeError('Final '+stage+' failed')
 dump({'complete':True,'completed':done,'results':str(R/'results.json')})
except Exception as e:dump({'error':repr(e),'traceback':traceback.format_exc()});raise
