import os,json,time,subprocess,traceback
from pathlib import Path
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';G=B/'data/AttentionPIT/expanded01/neighbors01';O=B/'runs/AttentionPIT/expanded01/neighbors01'
def free():return {int(z.split(',')[0]):int(z.split(',')[1]) for z in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
try:
 for key,family,gpus,py in [('Qwen3','Qwen3',[6],'attentionpit'),('Mistral','Mistral7B',[2,3,4,5],'attentionpit'),('gpt-oss-20b','GPToss20B',[6,7,2,3,4,5],'attentionpit'),('deepseek-v2-lite-chat','DeepSeekV2Lite',[6,7],'attentionpit'),('kimi-vl-a3b-instruct','KimiVLA3B',[6,7],'attentionpit-kimi')]:
  if (G/(family+'_title_features.npy')).exists():continue
  while True:
   f=free();limits=([18000] if key=='Qwen3' else [4400]*4 if key=='Mistral' else [19000,19000,5500,5500,4400,4400] if key=='gpt-oss-20b' else [19000,19000])
   if all(f[g]>=lim for g,lim in zip(gpus,limits)):break
   time.sleep(10)
  with open(O/(family+'_title.log'),'a') as log:
   p=subprocess.Popen([str(B/'venvs'/py/'bin/python'),'-u',str(C/'neighbor_title_encode.py'),key],cwd=C,env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',USE_TF='0',USE_FLAX='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'),stdout=log,stderr=subprocess.STDOUT);(O/'title_state.json').write_text(json.dumps({'family':family,'pid':p.pid,'gpus':gpus},indent=2))
   if p.wait():raise RuntimeError(family+' title encoding failed')
 (O/'title_state.json').write_text(json.dumps({'complete':True}))
except Exception as e:(O/'title_failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2));raise
