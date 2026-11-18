"""Batched revision retrieval, same frozen cohort and strict historical cutoffs."""
import os,json,subprocess,hashlib,datetime as dt,urllib.request,urllib.parse,time,shlex
B='/home/home_new/qsmx';D=os.environ.get('ATTENTION_PIT_DATA_ROOT',B+'/data/AttentionPIT/pilot01');R=os.environ.get('ATTENTION_PIT_RUN_ROOT',B+'/runs/AttentionPIT/pilot01');HOST='10.15.196.230';last=0

def rpc(code,input=None):
 p=subprocess.run(['ssh',HOST,'python3 -c '+shlex.quote(code)],input=input,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[-2000:])
 return p.stdout

def get(p):return json.loads(rpc('from pathlib import Path;p=Path('+repr(p)+');print(p.read_text() if p.exists() else "null")'))
def put(p,z):rpc('from pathlib import Path;import sys;p=Path('+repr(p)+');p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(sys.stdin.read());q.replace(p)',json.dumps(z,ensure_ascii=False))
def http(q):
 global last
 u='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',formatversion='2',maxlag=5,**q))
 for attempt in range(8):
  time.sleep(max(0,8-(time.monotonic()-last)));last=time.monotonic()
  try:
   with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'WebAttentionPITResearch/0.1 (batched historical revision research; sequential requests)'}),timeout=60) as f:z=json.load(f)
   if 'error' in z:raise RuntimeError(str(z['error']))
   return z,u
  except urllib.error.HTTPError as e:
   if e.code not in [429,503,502,504]:raise
   delay=max(60,int(e.headers.get('Retry-After','60')));print('BACKOFF',e.code,delay,flush=True);time.sleep(delay)
  except Exception:
   if attempt==7:raise
   time.sleep(15)
 raise RuntimeError('HTTP retries exhausted')
c=get(D+'/cohort.json');by={}
for z in c['rows']:by.setdefault(z['pageid'],[]).append(z)
completed=0
for pageid,rows in by.items():
 missing=[]
 for z in rows:
  key=hashlib.sha256((str(pageid)+'|'+z['origin']).encode()).hexdigest()[:24]
  if get(D+'/snapshots/'+key+'.json') is None:missing.append((z,key))
 if not missing:continue
 latest=max(z['origin'] for z,k in missing);earliest=min(z['origin'] for z,k in missing);start=(dt.datetime.fromisoformat(latest.replace('Z','+00:00'))-dt.timedelta(seconds=1)).isoformat().replace('+00:00','Z');metadata=[];cont={}
 while True:
  z,url=http(dict(prop='revisions',pageids=pageid,rvprop='ids|timestamp',rvstart=start,rvdir='older',rvlimit=500,**cont));page=z['query']['pages'][0];revs=page.get('revisions',[]);metadata.extend(revs)
  if any(r['timestamp']<earliest for r in revs) or 'continue' not in z:break
  cont=z['continue']
 chosen={}
 for row,key in missing:
  candidates=[r for r in metadata if r['timestamp']<row['origin']];chosen[key]=max(candidates,key=lambda r:r['timestamp']) if candidates else None
 ids=sorted({r['revid'] for r in chosen.values() if r});content={};contenturl=url
 if ids:
  z,contenturl=http(dict(prop='revisions',revids='|'.join(map(str,ids)),rvprop='ids|timestamp|content',rvslots='main'))
  for page in z['query']['pages']:
   for r in page.get('revisions',[]):content[r['revid']]=dict(r,title_at_query=page.get('title'))
 for row,key in missing:
  r=chosen[key];v=content.get(r['revid'],{}) if r else {};text=v.get('slots',{}).get('main',{}).get('content');record={'key':key,'pageid':pageid,'title_at_query':v.get('title_at_query',row['title']),'origin':row['origin'],'revid':r['revid'] if r else None,'revision_time':r['timestamp'] if r else None,'wikitext':text,'available':text is not None,'request_url':contenturl,'asof_query_url':url}
  assert not record['available'] or record['revision_time']<row['origin']
  put(D+'/snapshots/'+key+'.json',record);completed+=1
  print('SNAPSHOT',completed,row['split'],row['kind'],row['lead'],row['title'],record['available'],flush=True)
 put(R+'/collection_state.json',{'new_snapshots':completed,'last_page':rows[0]['title'],'method':'batch metadata then exact revision contents;8-second spacing'})
put(D+'/ready.json',{'rows':len(c['rows']),'completed':True});print('COLLECTION COMPLETE',flush=True)
