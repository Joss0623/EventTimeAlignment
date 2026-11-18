import os,json,time,subprocess,sys
from pathlib import Path
B=Path('/home/home_new/qsmx');R=B/'runs/AttentionPIT/pilot01';C=B/'pycodes/AttentionPIT'
keys=['gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct'];done=[]
while len(done)<len(keys):
 for key in keys:
  if key in done:continue
  if (R/(key+'_selection.json')).exists():done.append(key);continue
  if not (R/(key+'_verified.json')).exists():continue
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
  free=sorted([(int(l.split(',')[1]),int(l.split(',')[0])) for l in lines],reverse=True)
  if len([v for v in free if v[0]>=21000])<3:continue
  gpu=','.join(str(v[1]) for v in free[:3]);env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0')
  py=B/'venvs'/('attentionpit-kimi' if key.startswith('kimi') else 'attentionpit')/'bin/python'
  print('START',key,gpu,flush=True)
  (R/'family_state.json').write_text(json.dumps({'active':key,'gpus':gpu,'completed':done,'seed':2024}))
  with (R/(key+'_evaluate.log')).open('a') as log:r=subprocess.run([str(py),'-u',str(C/(key+'_evaluate.py')),key],env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:
   (R/'family_failure.json').write_text(json.dumps({'model':key,'exit_code':r.returncode}));sys.exit(r.returncode)
  done.append(key);print('COMPLETE',key,flush=True)
 time.sleep(20)
(R/'family_state.json').write_text(json.dumps({'completed':done,'complete':True,'seed':2024}))
