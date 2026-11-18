"""Fixed-target historical revision replay. Run via SSH pipe on network-enabled host."""
import json,subprocess,hashlib,datetime as dt,urllib.request,urllib.parse,time,re,sys
B='/home/home_new/qsmx';D=B+'/data/AttentionPIT/pilot01';R=B+'/runs/AttentionPIT/pilot01';HOST='10.15.196.230'
def rpc(code,input=None):
 p=subprocess.run(['ssh',HOST,'python3 -c '+__import__('shlex').quote(code)],input=input,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[-1500:])
 return p.stdout
def put(path,z):rpc('from pathlib import Path;import sys;p=Path('+repr(path)+');p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(sys.stdin.read());q.replace(p)',json.dumps(z,ensure_ascii=False))
def get(path):
 v=rpc('from pathlib import Path;p=Path('+repr(path)+');print(p.read_text() if p.exists() else "null")');return json.loads(v)
# Freeze selection by hash before retrieving snapshots or examining outcomes.
prep=r'''
import json,hashlib,datetime as dt
from pathlib import Path
b=Path('/home/home_new/qsmx');a=json.load(open(b/'data/EventTail/pilot01/aligned_events.json'));out=[];seen=set()
for split,n,accept in [('train',20,lambda x:x<'2024-09-24'),('val',8,lambda x:'2024-10-01'<=x<='2024-12-24'),('test',12,lambda x:x>='2025-01-01')]:
 pool=sorted([z for z in a if accept(z['spike_date'])],key=lambda z:hashlib.sha256(('2024|'+z['id']).encode()).hexdigest())
 count=0
 for z in pool:
  if z['pageid'] in seen:continue
  seen.add(z['pageid']);out.append({'event_id':z['id'],'pageid':z['pageid'],'title':z['title'],'event_date':z['spike_date'],'split':split,'raw_sha256':z['pageviews_sha256']});count+=1
  if count==n:break
 assert count==n
wanted={z['raw_sha256'] for z in out};raw={}
for f in (b/'data/EventTail/pilot01/traffic').glob('*.json'):
 v=json.load(open(f))
 if v.get('sha256') in wanted:raw[v['sha256']]=v
rows=[]
for z in out:
 v=raw[z['raw_sha256']];daily={x['timestamp'][:8]:x['views'] for x in v['items']};eventday=dt.date.fromisoformat(z['event_date'])
 for kind,day in [('event',eventday),('calendar_control',eventday-dt.timedelta(days=28))]:
  target=[daily.get((day+dt.timedelta(days=j)).strftime('%Y%m%d')) for j in range(7)]
  for lead in [7,3,1]:
   origin=day-dt.timedelta(days=lead);hist=[daily.get((origin-dt.timedelta(days=j)).strftime('%Y%m%d')) for j in range(28,0,-1)]
   rows.append(dict(z,kind=kind,target_start=day.isoformat(),lead=lead,origin=origin.isoformat()+'T00:00:00Z',history28=hist,target7=target,numeric_complete=None not in hist+target,raw_url=v['request_url']))
print(json.dumps({'seed':2024,'events':out,'rows':rows},ensure_ascii=False))
'''
cohort=get(D+'/cohort.json')
if cohort is None:cohort=json.loads(rpc(prep));put(D+'/cohort.json',cohort)
protocol={'seed':2024,'n_events':40,'n_calendar_controls':40,'leads':[7,3,1],'fixed_target':'seven daily pageviews starting target_start; identical target at all three forecast origins','origins':'00UTC target_start minus lead; numeric history ends previous day. Strict point-in-time daily replay, not claim of contemporaneous API publication latency','text_source':'target Wikipedia article historical revision strictly before origin; raw wikitext only, no live transclusion expansion or current linked news bodies','text_scope':'Wikipedia article information, not all public news. First observed in source is not first public disclosure globally.','controls':'same page at event_date minus28 calendar days; not screened using future outcomes and may themselves contain bursts; controls remain in same split as parent event','split':'20train/8val/12test independent pages selected by fixed seed hash, before examining outcomes; each page has both kinds and three leads','primary':'same-lead numeric vs text risks, not attributing numeric-history advances to text','LLM':'existing frozen Qwen3-8B embeddings with identical supervised readout to lexical features; this is an embedding-based LLM method, not a reasoning-agent capability test','status':'collecting'}
put(R+'/protocol.json',protocol)
last=0;count=0;missing=0
for row in cohort['rows']:
 key=hashlib.sha256((str(row['pageid'])+'|'+row['origin']).encode()).hexdigest()[:24];path=D+'/snapshots/'+key+'.json';old=get(path)
 if old is not None:continue
 q={'action':'query','format':'json','formatversion':'2','prop':'revisions','pageids':row['pageid'],'rvprop':'ids|timestamp|content','rvslots':'main','rvstart':(dt.datetime.fromisoformat(row['origin'].replace('Z','+00:00'))-dt.timedelta(seconds=1)).isoformat().replace('+00:00','Z'),'rvdir':'older','rvlimit':1,'maxlag':5};url='https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode(q)
 for attempt in range(8):
  delay=max(0,8-(time.monotonic()-last))
  if delay:time.sleep(delay)
  try:
   last=time.monotonic()
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'WebAttentionPITResearch/0.1 (historical snapshot research; sequential requests)'}),timeout=60) as response:z=json.load(response)
   if 'error' in z:raise RuntimeError(str(z['error']))
   break
  except urllib.error.HTTPError as e:
   if e.code not in [429,503,502,504]:raise
   wait=max(60,int(e.headers.get('Retry-After','60')));print('BACKOFF',e.code,wait,flush=True);time.sleep(wait)
  except Exception:
   if attempt==7:raise
   time.sleep(15)
 else:raise RuntimeError('HTTP retries exhausted')
 page=z['query']['pages'][0];revs=page.get('revisions',[]);r=revs[0] if revs else {};text=r.get('slots',{}).get('main',{}).get('content')
 record={'key':key,'pageid':row['pageid'],'title_at_query':page.get('title'),'origin':row['origin'],'revid':r.get('revid'),'revision_time':r.get('timestamp'),'wikitext':text,'available':text is not None,'request_url':url}
 if record['available']:assert record['revision_time']<row['origin']
 else:missing+=1
 put(path,record);count+=1;print('SNAPSHOT',count,'/',len(cohort['rows']),row['split'],row['kind'],row['lead'],record['available'],flush=True)
 if count%10==0:put(R+'/collection_state.json',{'new_snapshots':count,'missing':missing,'last_origin':row['origin'],'last_title':row['title']})
put(D+'/ready.json',{'rows':len(cohort['rows']),'completed':True});print('COLLECTION COMPLETE',flush=True)
