import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import json,re,hashlib,datetime as dt,html,collections
from run import load,snapshot_key,dump,B,D,R
rows,x,y=load();G=D/'neighbors01';O=R/'neighbors01';G.mkdir(exist_ok=True);O.mkdir(exist_ok=True);cache=G/'traffic';cache.mkdir(exist_ok=True)
def normalize(s):
 s=html.unescape(s).replace('_',' ').strip().lstrip(':').split('#')[0].strip();return s[:1].upper()+s[1:]
def links(text,selftitle):
 text=re.sub(r'<!--.*?-->|<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>',' ',text,flags=re.S|re.I)
 for _ in range(15):text=re.sub(r'\{\{[^{}]*\}\}',' ',text)
 text=re.split(r'(?im)^==+\s*(?:References|External links|See also|Notes|Further reading)\s*==+',text)[0]
 out=[]
 for raw in re.findall(r'\[\[([^|\]#]+)(?:[^\]]*)\]\]',text):
  t=normalize(raw)
  if not t or t==normalize(selftitle) or t in out:continue
  if re.match(r'(?i)^(File|Image|Category|Template|Help|Wikipedia|Portal|Special|Talk|User|Draft|MediaWiki|Module):',t):continue
  if re.fullmatch(r'\d{1,4}(?:s| BC| AD)?',t) or re.match(r'(?i)^(List of |January|February|March|April|May|June|July|August|September|October|November|December)(?:\b)',t):continue
  out.append(t)
  if len(out)==4:break
 return out
specs=[];by={};need=collections.defaultdict(set)
for i,z in enumerate(rows):
 key=(z['event_id'],z['kind']);by.setdefault(key,[]).append(i)
for key,ids in by.items():
 early=next(i for i in ids if rows[i]['lead']==7);z=rows[early];snap=json.load(open(D/'snapshots'/(snapshot_key(z)+'.json')));ns=links(snap['wikitext'] or '',z['title']);spec={'event_id':key[0],'kind':key[1],'pageid':z['pageid'],'title':z['title'],'graph_origin':z['origin'],'source_revid':snap['revid'],'source_revision_time':snap['revision_time'],'neighbors':ns,'row_indices':ids};assert not snap['available'] or snap['revision_time']<z['origin'];specs.append(spec)
 for i in ids:
  origin=dt.date.fromisoformat(rows[i]['origin'][:10]);days=[(origin-dt.timedelta(days=d)).isoformat() for d in range(28,0,-1)]
  for t in ns:need[t].update(days)
raw=collections.defaultdict(dict)
for f in (B/'data/EventTail/pilot01/traffic').glob('*.json'):
 d=json.load(open(f));t=normalize(d['title_at_request'])
 if t not in need:continue
 for v in d.get('items',[]):raw[t][dt.datetime.strptime(v['timestamp'][:8],'%Y%m%d').date().isoformat()]=int(v['views'])
requests=[];reused=0
for t,days in sorted(need.items()):
 key=hashlib.sha256(t.encode()).hexdigest()[:24];path=cache/(key+'.json');requests.append({'title':t,'key':key,'start':min(days),'end':max(days),'required_dates':sorted(days)})
 if path.exists():continue
 if all(day in raw[t] for day in days):dump(path,{'title':t,'daily':raw[t],'source':'reused original API cache','available':True});reused+=1
protocol={'seed':2024,'graph':'directed1hop star, neighbor-to-target, fixed per target window from earliest7day historical article revision','neighbors':4,'selection':'first4 distinct mainspace wikilinks in archived body after removing comments,refs,templates; omit self,namespace,date/list links; no ranking using traffic or future outcomes','information':'same28days ending origin-1 for every neighbor; no neighbor future values or labels','missing':'missing/unavailable neighbors masked, targets retained; successful historical AQS200 omitted dates recorded as zero per official API docs','conditions':['matched numerical MLP','neighbor MeanPool','one-layer learned graph attention','text only existing five frozen encoders','graph attention+each of five frozen encoders'],'numeric_reference':'same validation-selected Chronos2, same residual forecasting interface','budget':'same seed2024; min30/max150/patience20;validation pooledWAPE only; retained existing baseline predictions; no test-based search','goal':'separate extra neighbor information from graph aggregation and text information','api_policy':'one sequential AnalyticsAPI request at a time, >=1second between requests, retry429/503 afterRetryAfter>=60sec, no IP rotation'}
dump(G/'graph_specs.json',specs);dump(G/'requests.json',requests);dump(O/'protocol.json',protocol);summary={'target_windows':len(specs),'rows':len(rows),'neighbors_per_window_histogram':{str(k):sum(len(s['neighbors'])==k for s in specs) for k in range(5)},'unique_neighbor_titles':len(need),'requests_reused_from_original_cache':reused,'to_collect':sum(not(cache/(z['key']+'.json')).exists() for z in requests)};dump(O/'preparation.json',summary);print(json.dumps(summary,indent=2));print('FIRST',specs[0])
