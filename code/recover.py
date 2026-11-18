import os,time,subprocess,json,traceback
from pathlib import Path
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';R=B/'runs/AttentionPIT/expanded01';D=B/'data/AttentionPIT/expanded01';env=dict(os.environ,ATTENTION_PIT_DATA_ROOT=str(D),ATTENTION_PIT_RUN_ROOT=str(R),PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
def dump(v):(R/'recovery_state.json').write_text(json.dumps(v,indent=2))
try:
 for key,fn,py in [('gpt-oss-20b','gpt-oss-20b_evaluate.py','attentionpit'),('deepseek-v2-lite-chat','deepseek-v2-lite-chat_evaluate.py','attentionpit'),('kimi-vl-a3b-instruct','kimi-vl-a3b-instruct_evaluate.py','attentionpit-kimi'),('mistral','mistral.py','attentionpit')]:
  if (D/(key+'_embeddings.npz')).exists():continue
  while True:
   free={int(v.split(',')[0]):int(v.split(',')[1]) for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};g=sorted([k for k,v in free.items() if v>=(5500 if key=='gpt-oss-20b' else 21000)],key=lambda k:free[k],reverse=True);g=g[:4] if key=='gpt-oss-20b' else g[:1 if key=='mistral' else 2]
   if len(g)>=(4 if key=='gpt-oss-20b' else 1 if key=='mistral' else 2):break
   dump({'stage':'waiting_for_gpu','model':key});time.sleep(20)
  cmd=[str(B/'venvs'/py/'bin/python'),'-u',str(C/fn)]+([key] if key!='mistral' else [])+['--embed-only']
  with (R/(key+'.log')).open('a') as log:
   p=subprocess.Popen(cmd,cwd=C,env=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,g))),stdout=log,stderr=subprocess.STDOUT);dump({'stage':key,'pid':p.pid,'gpus':g})
   if p.wait():raise RuntimeError(key+' failed')
 dump({'stage':'full_neural_training'})
 with (R/'neural_readouts.log').open('a') as log:
  p=subprocess.Popen([str(B/'venvs/attentionpit/bin/python'),'-u',str(C/'neural_readouts.py'),'manage'],cwd=C,env=env,stdout=log,stderr=subprocess.STDOUT);dump({'stage':'full_neural_training','pid':p.pid})
  if p.wait():raise RuntimeError('Neural readouts failed')
 dump({'complete':True,'results':str(R/'neural_readouts/results.json')})
except Exception as e:dump({'error':repr(e),'traceback':traceback.format_exc()});raise
