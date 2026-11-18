import os,json,time,subprocess,traceback
from pathlib import Path
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';D=B/'data/AttentionPIT/alert01';R=B/'runs/AttentionPIT/alert01';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def run(key,gpus,py='attentionpit',fn='alert_encode.py'):
 with open(R/(key+'_encode.log'),'a') as log:
  p=subprocess.Popen([str(B/'venvs'/py/'bin/python'),'-u',str(C/fn)]+([key] if fn=='alert_encode.py' else []),cwd=C,env=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus))),stdout=log,stderr=subprocess.STDOUT)
  (R/'encoding_state.json').write_text(json.dumps({'key':key,'pid':p.pid,'gpus':gpus},indent=2))
  if p.wait():raise RuntimeError(key+' encoder failed')
def free():return {int(s.split(',')[0]):int(s.split(',')[1]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
try:
 run('TFIDF',[])
 while any(free()[g]<4600 for g in [2,3,4,5]):time.sleep(10)
 run('Mistral',[2,3,4,5])
 while not (D/'Qwen3_features.npy').exists() or not(D/'Chronos2_features.npy').exists():
  if list(R.glob('*_encoding_failure.json')):raise RuntimeError('Prerequisite encoder failed; inspect files')
  time.sleep(10)
 for key in ['gpt-oss-20b','deepseek-v2-lite-chat','kimi-vl-a3b-instruct']:
  if (D/(key+'_features.npy')).exists():continue
  while True:
   f=free()
   if f[6]>=21000 and f[7]>=21000 and (key!='gpt-oss-20b' or f[2]>=5500 and f[3]>=5500):break
   time.sleep(10)
  run(key,[6,7,2,3] if key=='gpt-oss-20b' else [6,7],py='attentionpit-kimi' if key=='kimi-vl-a3b-instruct' else 'attentionpit')
 (R/'encoding_state.json').write_text(json.dumps({'complete':True}))
except Exception as e:(R/'encoding_failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise
