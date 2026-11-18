import os,json,time,subprocess,traceback
from pathlib import Path
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';O=B/'runs/AttentionPIT/expanded01/posttrain01';PY=B/'venvs/attentionpit/bin/python'
def write(name,value):
 p=O/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
def main():
 claim=json.load(open(O/'pair_allocation.json'))
 for family,mode,lr in claim['configs']:
  dest=O/f'{family}_{mode}_lr{lr:g}'
  if (dest/'result.json').exists():continue
  # Adopt only this exact delegated job if a prior manager exited.
  existing=None
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:
    cmd=(proc/'cmdline').read_bytes().decode().split('\0')
    if str(C/'posttrain_lora.py') not in cmd or 'worker' not in cmd:continue
    if cmd[cmd.index('--family')+1]==family and cmd[cmd.index('--mode')+1]==mode and float(cmd[cmd.index('--lr')+1])==lr:existing=int(proc.name)
   except (FileNotFoundError,ValueError,IndexError,PermissionError):pass
  if existing:
   while Path('/proc',str(existing)).exists():
    if Path('/proc',str(existing),'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
    write('pair_state.json',{'job':[family,mode,lr],'pid':existing,'gpus':[2,3],'adopted':True});time.sleep(20)
   if not (dest/'result.json').exists():raise RuntimeError('Adopted pair worker failed')
   continue
  stable=0
  while stable<3:
   free={int(z.split(',')[0]):int(z.split(',')[1]) for z in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()};minimum=4500 if mode=='frozen' else 5500;stable=stable+1 if all(free[g]>=minimum for g in [2,3]) else 0;write('pair_state.json',{'stage':'waiting_stable_pair_memory','gpus':[2,3],'job':[family,mode,lr],'free_MiB':{g:free[g] for g in [2,3]},'consecutive_checks':stable})
   if stable<3:time.sleep(10)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',ATTENTION_PIT_SHARD2='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',ATTENTION_PIT_DATA_ROOT=str(B/'data/AttentionPIT/expanded01'),ATTENTION_PIT_RUN_ROOT=str(B/'runs/AttentionPIT/expanded01'),USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
  with open(O/(dest.name+'_pair.log'),'a') as log:
   p=subprocess.Popen([str(PY),'-u',str(C/'posttrain_lora.py'),'worker','--family',family,'--mode',mode,'--lr',str(lr)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
   write('pair_state.json',{'stage':'running','pid':p.pid,'gpus':[2,3],'job':[family,mode,lr]});code=p.wait()
  if code or not (dest/'result.json').exists():raise RuntimeError('Pair worker failed '+str([family,mode,lr,code]))
 write('pair_complete.json',{'complete':True,'gpus_released':[2,3],'configs':claim['configs']})
if __name__=='__main__':
 try:main()
 except Exception as e:write('pair_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
