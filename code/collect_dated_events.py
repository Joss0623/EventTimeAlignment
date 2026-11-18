"""Historical scheduled-event collection. Execute in local memory; artifacts on server."""
import os,json,time,hashlib,datetime as dt,urllib.request,urllib.parse,urllib.error,subprocess,shlex,traceback,socket
HOST='10.15.196.230';BASE='/home/home_new/qsmx';D=BASE+'/data/AttentionPIT/dated_events01';R=BASE+'/runs/AttentionPIT/dated_events01';last=0
GET_CACHE={};EXISTS_CACHE={}
def rpc(code,payload=None):
 p=subprocess.run(['ssh','-o','ConnectTimeout=15',HOST,'python3 -c '+shlex.quote(code)],input=None if payload is None else json.dumps(payload,ensure_ascii=False),text=True,capture_output=True)
 if p.returncode:raise RuntimeError('SSH RPC '+p.stderr[-1000:])
 return json.loads(p.stdout) if p.stdout.strip() else None
def get(path):
 if path not in GET_CACHE:GET_CACHE[path]=rpc('import json;from pathlib import Path;p=Path('+repr(path)+');print(p.read_text() if p.exists() else "null")')
 return GET_CACHE[path]
def put(path,z):
 GET_CACHE[path]=z;EXISTS_CACHE[path]=True
 rpc('import sys;from pathlib import Path;p=Path('+repr(path)+');p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(sys.stdin.read());q.replace(p)',z)
def exists(paths):
 missing=[p for p in paths if p not in EXISTS_CACHE]
 if missing:EXISTS_CACHE.update(rpc('import json,sys;from pathlib import Path;print(json.dumps({p:Path(p).exists() for p in json.load(sys.stdin)}))',missing))
 return {p:EXISTS_CACHE[p] for p in paths}
def put_many(records):rpc('import json,sys;from pathlib import Path;records=json.load(sys.stdin);'+chr(10)+'for path,obj in records.items():\n p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(obj,ensure_ascii=False));q.replace(p)',records)
def http(url):
 global last
 for attempt in range(8):
  wait=max(0,8-(time.monotonic()-last))
  if wait:time.sleep(wait)
  last=time.monotonic()
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'WebAttentionPITResearch/0.2 (point-in-time scheduled-election research; sequential eight-second requests)'}),timeout=90) as f:z=json.load(f)
   if 'error' in z:
    if z['error'].get('code')=='maxlag':time.sleep(60);continue
    raise RuntimeError(z['error'])
   return z
  except urllib.error.HTTPError as e:
   if e.code==404:return {'missing':True,'http_status':404,'url':url}
   if e.code not in [429,500,502,503,504]:raise
   pause=max(60,int(e.headers.get('Retry-After','60')));print('HTTP_BACKOFF',e.code,pause,flush=True)
   for _ in range((pause+29)//30):time.sleep(min(30,pause));pause-=30
  except (TimeoutError,urllib.error.URLError):
   if attempt==7:raise
   time.sleep(15)
 raise RuntimeError('HTTP retry budget exhausted '+url)
def wiki(**q):
 u='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',formatversion=2,maxlag=5,**q));return http(u),u
def key(pid,origin):return hashlib.sha256((str(pid)+'|'+origin).encode()).hexdigest()[:24]
def targets(e):
 day=dt.date.fromisoformat(e['event_date']);out=[]
 for kind,t in [('event',day-dt.timedelta(days=6)),('event',day-dt.timedelta(days=3)),('event',day),('calendar_control',day-dt.timedelta(days=28))]:
  for lead in [7,3,1]:
   origin=t-dt.timedelta(days=lead);out.append(dict(e,kind=kind,phase_offset=(t-day).days,target_start=str(t),lead=lead,origin=str(origin)+'T00:00:00Z'))
 return out
def main():
 frame=get(D+'/event_frame.json');resolved=get(D+'/resolved_frame.json')
 put(R+'/collector.json',{'pid':os.getpid(),'hostname':socket.gethostname(),'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'source':'local memory, remote artifacts only'})
 if resolved is None:
  mapping={};old={}
  for i in range(0,len(frame),40):
   part=frame[i:i+40];z,u=wiki(titles='|'.join(v['title'] for v in part),prop='info',redirects=1)
   pages={v['title']:v for v in z['query']['pages']};alias={v['from']:v['to'] for v in z['query'].get('normalized',[])+z['query'].get('redirects',[])}
   for e in part:
    title=e['title'];seen=set()
    while title in alias and title not in seen:seen.add(title);title=alias[title]
    page=pages.get(title)
    mapping[e['title']]=None if not page or page.get('missing') else {'pageid':page['pageid'],'canonical_title':page['title'],'resolution_url':u}
   put(D+'/title_resolution.json',mapping)
  previous=get(BASE+'/data/AttentionPIT/expanded01/cohort.json');oldtest={v['pageid'] for v in previous['rows'] if v['split']=='test'}
  records=[]
  for e in frame:
   v=mapping[e['title']]
   if v:records.append(dict(e,**v,old_test_overlap=v['pageid'] in oldtest))
  unique={}
  for e in sorted(records,key=lambda e:(e['event_date'],e['title'])):
   if e['pageid'] not in unique:unique[e['pageid']]=e
  resolved=sorted(unique.values(),key=lambda e:hashlib.sha256(('2024|frame|'+e['event_id']).encode()).hexdigest());put(D+'/resolved_frame.json',resolved)
 print('RESOLVED',len(resolved),{s:sum(e['split']==s for e in resolved) for s in ['train','val','test']},flush=True)
 # Reuse identical historical snapshots from the previous cohort; never copy current text.
 planned=[(row,key(e['pageid'],row['origin'])) for e in resolved for row in targets(e)]
 meta_paths=[D+'/revision_metadata/'+str(e['pageid'])+'.json' for e in resolved]
 GET_CACHE.update(rpc('import json,sys;from pathlib import Path;print(json.dumps({p:json.load(open(p)) if Path(p).exists() else None for p in json.load(sys.stdin)}))',meta_paths))
 reuse=rpc('import json,sys;from pathlib import Path;paths=json.load(sys.stdin);out={};'+chr(10)+'for k in paths:\n p=Path("/home/home_new/qsmx/data/AttentionPIT/expanded01/snapshots")/(k+".json");q=Path('+repr(D+'/snapshots')+')/(k+".json");\n if not q.exists() and p.exists():q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());out[k]=True'+chr(10)+'print(json.dumps(out))',[k for row,k in planned])
 print('REUSED_SNAPSHOTS',len(reuse),flush=True)
 exists([D+'/snapshots/'+k+'.json' for row,k in planned])
 metadata={}
 for j,e in enumerate(resolved):
  rows=targets(e);keys=[key(e['pageid'],v['origin']) for v in rows];present=exists([D+'/snapshots/'+k+'.json' for k in keys])
  if all(present.values()):continue
  path=D+'/revision_metadata/'+str(e['pageid'])+'.json';saved=get(path)
  if saved is None:
   earliest=min(v['origin'] for v in rows);latest=max(v['origin'] for v in rows);start=(dt.datetime.fromisoformat(latest.replace('Z','+00:00'))-dt.timedelta(seconds=1)).isoformat().replace('+00:00','Z');revisions=[];cont={};moves=[]
   while True:
    z,u=wiki(pageids=e['pageid'],prop='revisions',rvprop='ids|timestamp',rvstart=start,rvdir='older',rvlimit=500,list='logevents',letype='move',letitle=e['canonical_title'],lelimit=500,lestart=e['event_date']+'T23:59:59Z',leend=str(dt.date.fromisoformat(e['event_date'])-dt.timedelta(days=64))+'T00:00:00Z',leprop='ids|title|timestamp|details',**cont)
    if z.get('missing'):raise RuntimeError('Unexpected missing metadata')
    rr=z['query']['pages'][0].get('revisions',[]);revisions.extend(rr);moves.extend(z['query'].get('logevents',[]))
    if any(v['timestamp']<earliest for v in rr) or 'continue' not in z:break
    cont={k:v for k,v in z['continue'].items() if k in ['continue','rvcontinue']}
   saved={'pageid':e['pageid'],'revisions':revisions,'moves':moves,'request_url':u};put(path,saved)
  metadata[e['pageid']]=saved
  if j%10==0:put(R+'/collection_state.json',{'stage':'revision_metadata','completed_pages':j+1,'total_pages':len(resolved)});print('METADATA',j+1,'/',len(resolved),flush=True)
 chosen={}
 for e in resolved:
  rows=targets(e);keys=[key(e['pageid'],row['origin']) for row in rows];present=exists([D+'/snapshots/'+k+'.json' for k in keys])
  md=metadata.get(e['pageid']) or get(D+'/revision_metadata/'+str(e['pageid'])+'.json')
  for row,k in zip(rows,keys):
   if present[D+'/snapshots/'+k+'.json']:continue
   rr=[r for r in (md or {}).get('revisions',[]) if r['timestamp']<row['origin']];rev=max(rr,key=lambda v:v['timestamp']) if rr else None
   if rev is None:put(D+'/snapshots/'+k+'.json',{'key':k,'pageid':e['pageid'],'origin':row['origin'],'available':False,'wikitext':None,'revid':None,'revision_time':None,'request_url':(md or {}).get('request_url')})
   else:chosen.setdefault(rev['revid'],[]).append((row,k,rev))
 revids=sorted(chosen)
 for start in range(0,len(revids),40):
  ids=revids[start:start+40];z,u=wiki(revids='|'.join(map(str,ids)),prop='revisions',rvprop='ids|timestamp|content',rvslots='main');records={}
  for page in z['query']['pages']:
   for rev in page.get('revisions',[]):
    text=rev.get('slots',{}).get('main',{}).get('content')
    for row,k,meta in chosen.get(rev['revid'],[]):
     assert rev['timestamp']<row['origin']
     records[D+'/snapshots/'+k+'.json']={'key':k,'pageid':row['pageid'],'origin':row['origin'],'revid':rev['revid'],'revision_time':rev['timestamp'],'wikitext':text,'available':text is not None,'request_url':u,'title_at_query':page.get('title')}
  put_many(records);print('REVISION_CONTENT',min(start+40,len(revids)),'/',len(revids),flush=True);put(R+'/collection_state.json',{'stage':'revision_content','completed_revids':min(start+40,len(revids)),'total_revids':len(revids)})
 allrows=[];quality=[]
 for j,e in enumerate(resolved):
  path=D+'/traffic/'+str(e['pageid'])+'.json';v=get(path);day=dt.date.fromisoformat(e['event_date']);begin=day-dt.timedelta(days=64);end=day+dt.timedelta(days=6)
  if v is None:
   title=urllib.parse.quote(e['canonical_title'].replace(' ','_'),safe='');u='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/'+title+'/daily/'+begin.strftime('%Y%m%d00')+'/'+end.strftime('%Y%m%d00')
   z=http(u);v={'pageid':e['pageid'],'request_url':u,'response':z,'sha256':hashlib.sha256(json.dumps(z,sort_keys=True).encode()).hexdigest()};put(path,v)
  daily={p['timestamp'][:8]:p['views'] for p in v['response'].get('items',[])}
  md=metadata.get(e['pageid']) or get(D+'/revision_metadata/'+str(e['pageid'])+'.json');moves=(md or {}).get('moves',[]);hasmoves=bool(moves)
  for row in targets(e):
   origin=dt.date.fromisoformat(row['origin'][:10]);target=dt.date.fromisoformat(row['target_start'])
   h=[daily.get((origin-dt.timedelta(days=k)).strftime('%Y%m%d')) for k in range(28,0,-1)];y=[daily.get((target+dt.timedelta(days=k)).strftime('%Y%m%d')) for k in range(7)]
   allrows.append(dict(row,history28=h,target7=y,numeric_complete=None not in h+y and not hasmoves,move_window_flag=hasmoves,raw_url=v['request_url']))
  quality.append({'pageid':e['pageid'],'title':e['title'],'moves':moves,'traffic_missing':v['response'].get('missing',False)})
  if j%10==0:put(D+'/cohort_partial.json',{'seed':2024,'events':resolved,'rows':allrows});put(R+'/collection_state.json',{'stage':'pageviews','completed_pages':j+1,'total_pages':len(resolved)});print('PAGEVIEWS',j+1,'/',len(resolved),flush=True)
 cohort={'seed':2024,'events':resolved,'rows':allrows,'source':'historical snapshots and Wikimedia daily all-access user pageviews; fixed-calendar targets, no burst/outcome filtering'}
 put(D+'/cohort.json',cohort);put(D+'/quality.json',quality);put(D+'/ready.json',{'complete':True,'events':len(resolved),'rows':len(allrows),'numeric_complete':sum(v['numeric_complete'] for v in allrows)});put(R+'/collection_state.json',{'stage':'complete','events':len(resolved),'rows':len(allrows),'numeric_complete':sum(v['numeric_complete'] for v in allrows)});print('COLLECTION_COMPLETE',len(resolved),len(allrows),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  put(R+'/collection_failure.json',{'traceback':traceback.format_exc(),'time':dt.datetime.now(dt.timezone.utc).isoformat()});raise
