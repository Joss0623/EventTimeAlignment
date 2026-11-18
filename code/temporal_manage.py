import os,sys,json,subprocess,fcntl,time
from pathlib import Path
B=Path('/home/home_new/qsmx');VERSION=os.environ.get('ATTENTION_PIT_TEMPORAL_VERSION','temporal_semantics01');O=B/'runs/AttentionPIT/expanded01'/VERSION;O.mkdir(exist_ok=True)
lock=open(O/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,ATTENTION_PIT_DATA_ROOT=str(B/'data/AttentionPIT/expanded01'),ATTENTION_PIT_RUN_ROOT=str(B/'runs/AttentionPIT/expanded01'),CUDA_VISIBLE_DEVICES='1,2,3,4',PYTHONDONTWRITEBYTECODE='1')
for key in ['Qwen3','Mistral7B','GPToss20B','DeepSeekV2Lite','KimiVLA3B']:
 if (O/(key+'_complete.json')).exists():continue
 py=B/('venvs/attentionpit-kimi/bin/python' if key=='KimiVLA3B' else 'venvs/attentionpit/bin/python')
 with open(O/(key+'.log'),'a') as log:
  p=subprocess.Popen([str(py),'-u',str(B/'pycodes/AttentionPIT/temporal_semantics.py'),'encode','--model',key],env=env,stdout=log,stderr=subprocess.STDOUT)
  (O/'manager_state.json').write_text(json.dumps({'model':key,'pid':p.pid,'manager':os.getpid(),'started':time.time()}));print('START',key,p.pid,flush=True);rc=p.wait()
  if rc:raise RuntimeError((key,rc))
subprocess.run([str(B/'venvs/attentionpit/bin/python'),'-u',str(B/'pycodes/AttentionPIT/temporal_semantics.py'),'analyze'],env=env,check=True)
(O/'manager_state.json').write_text(json.dumps({'complete':True,'time':time.time()}))
