"""Exact eager GPT-oss attention, tiled along queries to bound workspace.
Retains the learned attention sinks, full keys, causal/sliding masks and dtype.
"""
import torch
from types import SimpleNamespace

def install():
 import transformers.models.gpt_oss.modeling_gpt_oss as g
 original=g.eager_attention_forward
 def tiled(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
  key=g.repeat_kv(key,module.num_key_value_groups);value=g.repeat_kv(value,module.num_key_value_groups);parts=[]
  for start in range(0,query.shape[-2],256):
   q=query[:,:,start:start+256,:];logits=torch.matmul(q,key.transpose(2,3))*scaling
   if attention_mask is not None:logits=logits+(attention_mask[...,start:start+q.shape[-2],:] if attention_mask.shape[-2]>1 else attention_mask)
   sinks=module.sinks.reshape(1,-1,1,1).expand(q.shape[0],-1,q.shape[-2],-1);combined=torch.cat([logits,sinks],dim=-1);combined=combined-combined.max(dim=-1,keepdim=True).values;prob=torch.softmax(combined,dim=-1,dtype=combined.dtype)[...,:-1];prob=torch.nn.functional.dropout(prob,p=dropout,training=module.training).to(value.dtype);parts.append(torch.matmul(prob,value))
  return torch.cat(parts,dim=2).transpose(1,2).contiguous(),None
 # Validate both full causal and sliding masks against the installed reference.
 gen=torch.Generator().manual_seed(2024);q=torch.randn(1,4,513,8,generator=gen);k=torch.randn(1,2,513,8,generator=gen);v=torch.randn(1,2,513,8,generator=gen);m=SimpleNamespace(num_key_value_groups=2,sinks=torch.randn(4,generator=gen),training=False);pos=torch.arange(513);errors=[]
 for window in [None,128]:
  valid=pos[None,:]<=pos[:,None]
  if window:valid=valid & (pos[None,:]>pos[:,None]-window)
  mask=torch.where(valid,0.,float('-inf'))[None,None,:,:];a=original(m,q,k,v,mask,8**-.5)[0];b=tiled(m,q,k,v,mask,8**-.5)[0];torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6);errors.append(float((a-b).abs().max()))
 g.eager_attention_forward=tiled
 print('ATTENTION WORKSPACE REPAIR: reference equivalence maxabs',errors,'query tile256; unchanged full context and sinks',flush=True)
