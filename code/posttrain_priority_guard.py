"""Restore only verified, voluntarily paused own jobs after LoRA priority window."""
import os,time,json,signal
from pathlib import Path
O=Path('/home/home_new/qsmx/runs/AttentionPIT/expanded01/posttrain01');C=Path('/home/home_new/qsmx/pycodes/AttentionPIT');journal=O/'paused_own_rift.json'
def alive_jobs():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   cmd=(p/'cmdline').read_bytes().decode().split('\0')
   if str(C/'posttrain_lora.py') in cmd and any(v in cmd for v in ['manager','worker','report']):return True
  except (FileNotFoundError,PermissionError):continue
 return False
def main():
 j=json.load(open(journal))
 if j.get('resumed'):return
 while True:
  reason='results_complete' if (O/'results.json').exists() else 'priority_window_24hours_elapsed' if time.time()-j['paused_unix']>86400 else 'all_posttrain_processes_exited' if not alive_jobs() else None
  if reason:break
  time.sleep(10)
 out=[]
 for v in sorted(j['paused'],key=lambda v:v['kind']=='queue'):
  p=Path('/proc')/str(v['pid'])
  try:
   if not p.exists():out.append({'pid':v['pid'],'status':'already_exited'});continue
   st=(p/'stat').read_text().rsplit(')',1)[1].split();args=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
   assert p.stat().st_uid==os.getuid() and st[19]==v['start_ticks'] and args==v['cmd'],'PID identity changed'
   if not v['was_stopped']:os.kill(v['pid'],signal.SIGCONT)
   out.append({'pid':v['pid'],'status':'resumed' if not v['was_stopped'] else 'preserved_original_stop'})
  except (FileNotFoundError,ProcessLookupError):out.append({'pid':v['pid'],'status':'exited_during_resume'})
 j.update(resumed=True,resumed_unix=time.time(),resume_reason=reason,resume_outcomes=out);q=journal.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2));q.replace(journal);print(json.dumps(j),flush=True)
if __name__=='__main__':main()
