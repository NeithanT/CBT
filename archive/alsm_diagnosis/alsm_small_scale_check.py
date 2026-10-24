"""Scaled-down ALSM (d=128, 2 core blocks, 3 loops) trained on 1-6 digit sums to test that the architecture and place indices learn addition.

Usage: python experiments/alsm_small_scale_check.py K_RAND_SHIFT STEPS
"""
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.alsm import ALSMConfig, AbacusLoopedMathGPT
from src.tokenizer import MathTokenizer
from src.arithmetic import format_addition_char_scratchpad
from training.trainer import encode_texts_to_batch
torch.set_num_threads(8); torch.manual_seed(0)
tok=MathTokenizer(); rng=random.Random(0)
def batch(n,dmax):
    texts=[]
    for _ in range(n):
        d=rng.randint(1,dmax); a,b=rng.randint(10**(d-1),10**d-1),rng.randint(10**(d-1),10**d-1)
        texts.append(format_addition_char_scratchpad(a,b))
    return encode_texts_to_batch(texts,tok,128,None,return_place_ids=True)
KS=int(sys.argv[1]); STEPS=int(sys.argv[2])
for ks,ds in ((KS,True),):
    cfg=ALSMConfig(vocab_size=tok.vocab_size,block_size=128,n_embd=128,n_head=4,n_core_layers=2,train_loops=3,eval_loops=3,k_rand_shift=ks,deep_supervision=ds,dropout=0.0)
    m=AbacusLoopedMathGPT(cfg); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    t0=time.time()
    for step in range(STEPS):
        x,y,p=batch(64,6); m.train()
        _,loss=m(x,place_ids=p,targets=y); opt.zero_grad(); loss.backward(); opt.step()
        if step%150==0 or step==STEPS-1: print(f"k={ks} deepsup={ds} step={step} loss={loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.eval(); ok=tot=0
    x,y,p=batch(200,6)
    with torch.no_grad(): lg,_=m(x,place_ids=p)
    mask=y!=-100; acc=((lg.argmax(-1)==y)&mask).sum().item()/mask.sum().item()
    print(f"  -> post-prompt next-char acc {acc:.3f}")
