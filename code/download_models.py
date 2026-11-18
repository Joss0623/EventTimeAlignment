import os,json,subprocess,sys,hashlib,time
from pathlib import Path
B=Path('/home/home_new/qsmx');M=B/'models/AttentionPIT';R=B/'runs/AttentionPIT/pilot01';key=sys.argv[1];v=json.load(open(M/'download_manifest.json'))[key];dest=M/key;dest.mkdir(exist_ok=True)
env=dict(os.environ,HF_ENDPOINT='https://hf-mirror.com',HF_HUB_DISABLE_XET='1',HF_HUB_DOWNLOAD_TIMEOUT='90',PYTHONDONTWRITEBYTECODE='1')
cmd=[str(B/'anaconda3/envs/py38_torch201/bin/hf'),'download',v['repo'],*v['files'],'--revision',v['revision'],'--local-dir',str(dest),'--max-workers','2']
for attempt in range(8):
 result=subprocess.run(cmd,env=env,capture_output=True,text=True)
 print(result.stdout,flush=True)
 if result.returncode==0:break
 print(result.stderr,flush=True)
 if attempt==7:raise RuntimeError('Download failed after 8 bounded retries')
 import re
 retry=re.search(r'Retry after (\d+) seconds',result.stderr)
 delay=max(60,int(retry.group(1)) if retry else min(600,60*2**attempt))
 print('BACKOFF',delay,flush=True);time.sleep(delay)
verified={}
for name,size in v['sizes'].items():
 p=dest/name
 if p.stat().st_size!=size:raise ValueError('Size mismatch '+name)
 if name in v['lfs_sha256']:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
  if h.hexdigest()!=v['lfs_sha256'][name]:raise ValueError('SHA mismatch '+name)
  verified[name]=h.hexdigest()
(R/(key+'_verified.json')).write_text(json.dumps({'repo':v['repo'],'revision':v['revision'],'verified_sha256':verified,'files':len(v['files']),'complete':True},indent=2))
print('VERIFIED',key,flush=True)
