import os
os.environ.update(ATTENTION_PIT_DATA_ROOT='/home/home_new/qsmx/data/AttentionPIT/expanded01',ATTENTION_PIT_RUN_ROOT='/home/home_new/qsmx/runs/AttentionPIT/expanded01',PYTHONDONTWRITEBYTECODE='1',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,sys,numpy as np,torch,hashlib,traceback
from transformers import AutoModel,AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from run import B,D,R,load,read_npz,snapshot_key,dump
G=D/'neighbors01';O=R/'neighbors01';KEY=sys.argv[1];FAMILY={'Qwen3':'Qwen3','Mistral':'Mistral7B','gpt-oss-20b':'GPToss20B','deepseek-v2-lite-chat':'DeepSeekV2Lite','kimi-vl-a3b-instruct':'KimiVLA3B'}[KEY]
def main():
 torch.set_num_threads(2);torch.manual_seed(2024);np.random.seed(2024);titles=json.load(open(G/'title_catalogue.json'));dest=G/(FAMILY+'_title_embeddings.npz');texts=['Wikipedia page: '+t for t in titles]
 if dest.exists():raw=np.load(dest)['embedding']
 else:
  path={'Qwen3':B/'models/ChronoGeom/qwen3-8b-b968826d','Mistral':B/'pycodes/WebBurst/weights/mistralai--Mistral-7B-Instruct-v0.2'}.get(KEY,B/'models/AttentionPIT'/KEY);tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=KEY=='kimi-vl-a3b-instruct');tok.padding_side='left'
  if tok.pad_token_id is None:tok.pad_token=tok.eos_token
  kwargs=dict(torch_dtype=torch.bfloat16,local_files_only=True,device_map='auto',max_memory={i:'19GiB' for i in range(torch.cuda.device_count())})
  if KEY=='gpt-oss-20b':
   from transformers import Mxfp4Config
   from gpt_attention import install
   install();kwargs.update(quantization_config=Mxfp4Config(dequantize=True),attn_implementation='eager',device_map={'embed_tokens':0,'rotary_emb':0,'norm':5,**{'layers.'+str(i):0 if i<8 else 1 if i<16 else 2 if i<18 else 3 if i<20 else 4 if i<22 else 5 for i in range(24)}})
  elif KEY=='Mistral':kwargs.update(attn_implementation='sdpa',device_map={'embed_tokens':0,'norm':3,'rotary_emb':0,**{'layers.'+str(i):i//8 for i in range(32)}})
  elif KEY=='kimi-vl-a3b-instruct':kwargs.update(trust_remote_code=True,attn_implementation='eager')
  else:kwargs.update(attn_implementation='eager' if KEY=='deepseek-v2-lite-chat' else 'sdpa')
  wrapper=AutoModel.from_pretrained(path,**kwargs).eval();m=wrapper.language_model.model if KEY=='kimi-vl-a3b-instruct' else wrapper;device=m.get_input_embeddings().weight.device;torch.cuda.empty_cache();parts=[]
  for start in range(0,len(titles),8):
   inp=tok(texts[start:start+8],padding=True,return_tensors='pt',add_special_tokens=False);inp={k:v.to(device) for k,v in inp.items()};inp['position_ids']=(inp['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
   with torch.inference_mode():h=m(**inp,use_cache=False).last_hidden_state;mask=inp['attention_mask'].to(h.device).unsqueeze(-1);v=(h.float()*mask).sum(1)/mask.sum(1).clamp_min(1);parts.append(v.cpu().numpy())
   if start%800==0:print(KEY,'TITLES',start,'/',len(titles),flush=True)
  raw=np.concatenate(parts);assert np.isfinite(raw).all();np.savez_compressed(dest,embedding=raw,titles=np.array(titles));del wrapper,m;torch.cuda.empty_cache()
 rows,x,y=load();tr=np.array([i for i,z in enumerate(rows) if z['split']=='train']);sel=json.load(open(R/'neural_readouts/selected_configs.json'))[FAMILY];dim=sel['dim']
 if KEY=='Qwen3':root=np.stack([read_npz(D/'representations'/(snapshot_key(z)+'.npz'))['embedding'] for z in rows])
 else:root=np.load(D/({'Mistral':'mistral'}.get(KEY,KEY)+'_embeddings.npz'))['embedding']
 root=root/np.maximum(np.linalg.norm(root,axis=1,keepdims=True),1e-8);pc=PCA(n_components=dim,random_state=2024).fit(root[tr]);raw=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8);q=pc.transform(raw);specs=json.load(open(G/'graph_specs.json'));training_titles={t for z in specs if rows[z['row_indices'][0]]['split']=='train' for t in z['neighbors']};ids=[i for i,t in enumerate(titles) if t in training_titles];sc=StandardScaler().fit(q[ids]);q=sc.transform(q);q=np.pad(q,((0,0),(0,32-dim))).astype(np.float32);assert np.isfinite(q).all();dump(O/(FAMILY+'_title_protocol.json'),{'seed':2024,'full_strings_same_for_every_model':True,'strings':'Wikipedia page: historical linked title; no current article body','titles':len(titles),'PCA_basis':'previously selected dimension, fitted only training-root article embeddings','neighbor_scaling':'training-neighbor-title embeddings only','training_neighbor_titles':len(ids),'dim':dim,'source_weights_reused':True});np.save(G/(FAMILY+'_title_features.npy'),q);print(FAMILY,'TITLE FEATURES COMPLETE',flush=True)
try:main()
except Exception as e:dump(O/(FAMILY+'_title_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
