import os,json,time,subprocess,itertools,traceback
from pathlib import Path
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';G=B/'data/AttentionPIT/expanded01/neighbors01';O=B/'runs/AttentionPIT/expanded01/neighbors01';PY=B/'venvs/attentionpit/bin/python';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
keys=['SMCB_Target','SMCB_TargetQwen3','Static_Neighbors','SMCB_Neighbors','SMCB_Metadata','SMCB_Qwen3ReadoutOnly','SMCB_Qwen3','SMCB_Mistral7B','SMCB_GPToss20B','SMCB_DeepSeekV2Lite','SMCB_KimiVLA3B'];queue=list(itertools.product(keys,[128,256],[3,7],[.0001,.0003,.001],['logMSE','logHuber']));active={};done=0
p=O/'protocol.json';v=json.load(open(p));v.update({'conditions':keys,'primary_module':'two temporal SMCB blocks, not a generic1layerGNN','configurations_per_condition':24,'total_configurations':264,'head_width':[128,256],'encoder_channels':'head_width/4','kernel':[3,7],'lr':[.0001,.0003,.001],'loss':['logMSE','logHuber'],'min_epochs':30,'max_epochs':150,'patience':20,'batch':64,'seed':2024,'selection':'checkpoint and config solely pooled validation WAPE; retained heldout events127/fresh115/calendar122','text_dims':'reuse previously validation-selected train-only PCA dimension per encoder; five frozen weights unchanged','counterfactuals':['nonlinked traffic matched using past-only scale at7day origin','seven-day-frozen neighbor histories','wrong words with recipient metadata unchanged','seven-day-frozen words'],'not_original_claim':'SMCB itself is borrowed from SGFNet and explicitly attributed; new hypothesis concerns Web inter-page attention information'});p.write_text(json.dumps(v,indent=2))
def name(t):k,w,ker,lr,loss=t;return f'{k}_w{w}_k{ker}_lr{lr:g}_{loss}'
def dump(x):
 p=O/'training_state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
class Adopted:
 def __init__(self,pid,task):self.pid=pid;self.task=task;self.returncode=None
 def poll(self):
  try:
   if (Path('/proc')/str(self.pid)/'stat').read_text().split()[2]!='Z':return None
  except FileNotFoundError:pass
  self.returncode=0 if (O/'trials'/name(self.task)/'result.json').exists() else 1;return self.returncode
class NoLog:
 def close(self):pass
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
  if not any(Path(v).name=='smcb_forecasting.py' for v in args) or 'worker' not in args:continue
  vals=lambda x:args[args.index(x)+1];task=(vals('--key'),int(vals('--width')),int(vals('--kernel')),float(vals('--lr')),vals('--loss'));ep=dict(z.split('=',1) for z in (proc/'environ').read_bytes().decode().split('\0') if '=' in z);gpu=int(ep['CUDA_VISIBLE_DEVICES']);active[gpu]=(Adopted(int(proc.name),task),task,NoLog())
  if task in queue:queue.remove(task)
 except (FileNotFoundError,PermissionError,ValueError,KeyError):continue
for task in list(queue):
 if (O/'trials'/name(task)/'result.json').exists():queue.remove(task);done+=1
try:
 while queue or active:
  if (G/'ready.json').exists() and not (G/'histories.npz').exists():
   with open(O/'assembly.log','a') as log:
    p=subprocess.run([str(PY),'-u',str(C/'neighbor_assemble.py')],cwd=C,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Neighbor history assembly failed')
  for gpu,(p,task,log) in list(active.items()):
   if p.poll() is not None:
    log.close()
    if p.returncode:raise RuntimeError('SMCB trial failed '+str(task))
    done+=1;del active[gpu]
  free={int(z.split(',')[0]):int(z.split(',')[1]) for z in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
  for gpu in [2,3,4,5]:
   if gpu in active or free[gpu]<3000:continue
   family={'SMCB_Qwen3ReadoutOnly':'Qwen3',**{'SMCB_'+k:k for k in ['Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B']}}
   picks=[t for t in queue if t[0].startswith('SMCB_Target') or (G/'histories.npz').exists() and (t[0] not in family or (G/(family[t[0]]+'_title_features.npy')).exists())]
   if not picks:continue
   task=picks[0];queue.remove(task);key,w,k,lr,loss=task;log=open(O/(name(task)+'.log'),'a');p=subprocess.Popen([str(PY),'-u',str(C/'smcb_forecasting.py'),'worker','--key',key,'--width',str(w),'--kernel',str(k),'--lr',str(lr),'--loss',loss],cwd=C,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT);active[gpu]=(p,task,log)
  dump({'complete':False,'completed':done,'queued':len(queue),'active':{str(g):{'pid':p.pid,'task':t} for g,(p,t,_) in active.items()},'waiting_for_neighbor_data':not(G/'histories.npz').exists(),'total':264,'seed':2024});time.sleep(10)
 with open(O/'analysis.log','a') as log:
  p=subprocess.run([str(PY),'-u',str(C/'smcb_forecasting.py'),'report'],cwd=C,env=env,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError('SMCB report failed')
 dump({'complete':True,'completed':done,'results':str(O/'results.json')})
except Exception as e:dump({'error':repr(e),'traceback':traceback.format_exc()});raise
