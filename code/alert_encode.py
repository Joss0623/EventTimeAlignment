import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
import sys,json,hashlib,time,traceback
from pathlib import Path
import numpy as np,torch
from transformers import AutoModel,AutoTokenizer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
B=Path('/home/home_new/qsmx');C=B/'pycodes/AttentionPIT';D=B/'data/AttentionPIT/alert01';R=B/'runs/AttentionPIT/alert01';KEY=sys.argv[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def aggregate(z,docs,rows,idx):
 out=np.zeros((len(rows),z.shape[1]),dtype=np.float32)
 for i,di in enumerate(idx):
  if di:
   weights=np.exp(-np.array([rows[i]['ordinal']-docs[j]['available'] for j in di])/7);out[i]=np.average(z[di],axis=0,weights=weights)
 return out
def main():
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024)
 docs=json.load(open(D/'documents.json'));rows=json.load(open(D/'rows.json'));idx=json.load(open(D/'document_indices.json'));tr=np.array([i for i,r in enumerate(rows) if r.get('original_split',r['split'])=='train']);train_docs=sorted({d for i in tr for d in idx[i]});texts=[d['text'] for d in docs]
 if KEY=='TFIDF':
  tf=TfidfVectorizer(min_df=2,max_features=12000,ngram_range=(1,2),sublinear_tf=True).fit([texts[i] for i in train_docs]);raw=tf.transform(texts);pc=TruncatedSVD(n_components=32,random_state=2024).fit(raw[train_docs]);small=pc.transform(raw)
 else:
  paths={'Qwen3':B/'models/ChronoGeom/qwen3-8b-b968826d','Mistral':B/'pycodes/WebBurst/weights/mistralai--Mistral-7B-Instruct-v0.2'};path=paths.get(KEY,B/'models/AttentionPIT'/KEY);cache=D/(KEY+'_documents');cache.mkdir(exist_ok=True)
  files=[cache/(hashlib.sha256(t.encode()).hexdigest()+'.npy') for t in texts]
  if not all(p.exists() for p in files):
   tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=KEY=='kimi-vl-a3b-instruct');kw=dict(torch_dtype=torch.bfloat16,local_files_only=True,device_map='auto',max_memory={i:'19GiB' for i in range(torch.cuda.device_count())})
   if KEY=='gpt-oss-20b':
    from transformers import Mxfp4Config
    from gpt_attention import install
    install();kw.update(quantization_config=Mxfp4Config(dequantize=True),attn_implementation='eager',device_map={'embed_tokens':0,'rotary_emb':0,'norm':3,**{'layers.'+str(i):0 if i<10 else 1 if i<20 else 2 if i<22 else 3 for i in range(24)}})
   elif KEY=='kimi-vl-a3b-instruct':kw.update(trust_remote_code=True,attn_implementation='eager')
   elif KEY=='Mistral':kw.update(attn_implementation='sdpa',device_map={'embed_tokens':0,'norm':3,'rotary_emb':0,**{'layers.'+str(i):i//8 for i in range(32)}})
   elif KEY=='deepseek-v2-lite-chat':kw.update(attn_implementation='eager')
   else:kw.update(attn_implementation='sdpa')
   wrapper=AutoModel.from_pretrained(path,**kw).eval();model=wrapper.language_model.model if KEY=='kimi-vl-a3b-instruct' else wrapper;device=model.get_input_embeddings().weight.device;torch.cuda.empty_cache()
   for i,(text,f) in enumerate(zip(texts,files)):
    if f.exists():continue
    ids=tok(text,add_special_tokens=False)['input_ids'];zs=[];parts=[ids[j:j+4096] for j in range(0,len(ids),4096)]
    for part in parts:
     with torch.inference_mode():h=model(input_ids=torch.tensor([part],device=device),use_cache=False).last_hidden_state;zs.append(h.float().mean(1).cpu().numpy()[0])
    z=np.average(zs,axis=0,weights=[len(p) for p in parts]).astype(np.float32);assert np.isfinite(z).all();np.save(f,z)
    if (i+1)%50==0:print(KEY,'DOC',i+1,'/',len(docs),flush=True)
   del model,wrapper;torch.cuda.empty_cache()
  raw=np.stack([np.load(p) for p in files]);raw/=np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8);pc=PCA(n_components=32,random_state=2024).fit(raw[train_docs]);small=pc.transform(raw)
 full=aggregate(small,docs,rows,idx);sc=StandardScaler().fit(full[tr]);full=sc.transform(full).astype(np.float32)
 # Pure text replacement: same recipient timing/type/length metadata, earlier donor of same category.
 linked=json.load(open(B/'data/EventTail/pilot01/event_candidates.json'));sets=[set(c['candidate_titles']) for c in linked];wrong=np.zeros_like(small);donors=[]
 for i,d in enumerate(docs):
  pool=[j for j,e in enumerate(docs) if e['available']<=d['available'] and i!=j and d['category']==e['category'] and not(sets[i]&sets[j])]
  j=min(pool,key=lambda j:(abs(len(texts[i])-len(texts[j])),docs[j]['id'])) if pool else -1;donors.append(j)
  if j>=0:wrong[i]=small[j]
 alt=sc.transform(aggregate(wrong,docs,rows,idx)).astype(np.float32)
 np.save(D/(KEY+'_mismatch_features.npy'),alt);dump(R/(KEY+'_encoding.json'),{'seed':2024,'documents':len(docs),'common_full_strings':True,'future_used_in_fit':False,'PCA_training_documents':len(train_docs),'pooling':'token mean; recency7day weighted document mean; 28day text history','donors':donors})
 # Ready file is last, after mismatch and audit are durably saved.
 np.save(D/(KEY+'_features.npy'),full);print(KEY,'ENCODING COMPLETE',flush=True)
try:main()
except Exception as e:dump(R/(KEY+'_encoding_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
