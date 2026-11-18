import json,subprocess,shlex,urllib.request,urllib.error,urllib.parse,time,datetime as dt,hashlib,email.utils,traceback
HOST='10.15.196.230';G='/home/home_new/qsmx/data/AttentionPIT/expanded01/neighbors01';O='/home/home_new/qsmx/runs/AttentionPIT/expanded01/neighbors01'
def rpc(code,inp=None):
 p=subprocess.run(['ssh',HOST,'python3 -c '+shlex.quote(code)],input=inp,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[-1500:])
 return p.stdout
def get(p):return json.loads(rpc('from pathlib import Path;p=Path('+repr(p)+');print(p.read_text() if p.exists() else "null")'))
def put(p,z):rpc('from pathlib import Path;import sys;p=Path('+repr(p)+');p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(".tmp");q.write_text(sys.stdin.read());q.replace(p)',json.dumps(z,ensure_ascii=False))
requests=get(G+'/requests.json');existing=set(json.loads(rpc('from pathlib import Path;import json;print(json.dumps([p.stem for p in Path('+repr(G+'/traffic')+').glob("*.json")]))')));last=time.monotonic()+52;new=0;unavailable=0
try:
 for request in requests:
  if request['key'] in existing:continue
  start=request['start'].replace('-','')+'00';end=request['end'].replace('-','')+'00';url='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/'+urllib.parse.quote(request['title'].replace(' ','_'),safe='')+'/daily/'+start+'/'+end
  z=None;missing=False
  for attempt in range(8):
   time.sleep(max(0,8-(time.monotonic()-last)));last=time.monotonic()
   try:
    req=urllib.request.Request(url,headers={'User-Agent':'WebAttentionPITResearch/0.2 (historical Wikipedia attention study; sequential requests) Python-urllib'})
    with urllib.request.urlopen(req,timeout=60) as response:z=json.load(response)
    break
   except urllib.error.HTTPError as e:
    if e.code==404:missing=True;break
    if e.code not in [429,503,502,504]:raise
    h=e.headers.get('Retry-After','60')
    try:wait=max(60,int(h))
    except ValueError:wait=max(60,int(email.utils.parsedate_to_datetime(h).timestamp()-time.time()))
    print('BACKOFF',e.code,wait,flush=True)
    deadline=time.monotonic()+wait
    while time.monotonic()<deadline:time.sleep(min(30,deadline-time.monotonic()))
   except (TimeoutError,urllib.error.URLError):
    if attempt==7:raise
    time.sleep(10)
  if z is None and not missing:raise RuntimeError('Retries exhausted '+request['title'])
  daily={};omitted=0
  if not missing:
   assert 'items' in z and isinstance(z['items'],list),str(z)[:200]
   for v in z['items']:daily[dt.datetime.strptime(v['timestamp'][:8],'%Y%m%d').date().isoformat()]=int(v['views'])
   current=dt.date.fromisoformat(request['start']);endday=dt.date.fromisoformat(request['end'])
   while current<=endday:
    k=current.isoformat()
    if k not in daily:daily[k]=0;omitted+=1
    current+=dt.timedelta(days=1)
  payload={'title':request['title'],'request_url':url,'start':request['start'],'end':request['end'],'available':not missing,'http404':missing,'daily':daily,'zero_omitted_days':omitted,'raw_response':z,'sha256':hashlib.sha256(json.dumps(z,sort_keys=True).encode()).hexdigest()};put(G+'/traffic/'+request['key']+'.json',payload);new+=1;unavailable+=missing
  if new%10==0:
   state={'new_requests':new,'existing_requests':len(existing),'total':len(requests),'complete':False,'unavailable_404':unavailable,'last_title':request['title']};put(O+'/collection_state.json',state);print('NEIGHBOR',new+len(existing),'/',len(requests),request['title'],'available',not missing,flush=True)
 put(G+'/ready.json',{'complete':True,'total_titles':len(requests)});put(O+'/collection_state.json',{'complete':True,'new_requests':new,'existing_requests':len(existing),'total':len(requests),'unavailable_404':unavailable});print('NEIGHBOR COLLECTION COMPLETE',flush=True)
except Exception as e:put(O+'/collection_failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
