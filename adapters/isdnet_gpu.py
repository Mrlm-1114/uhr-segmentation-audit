"""ISDNet GPU preprocessing endpoint at global scale 0.8.

The route ISDNet_unified_S_0.8 is distinct from the unchanged ISDNet_unified
and ISDNet_official_down_0.8. Model construction is handled by the caller.
CPU execution is restricted to explicitly marked synthetic QA.
"""
import contextlib
from pathlib import Path
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

ROUTE_NAME='ISDNet_unified_S_0.8'
SCALE=0.8
MEAN=(123.675,116.28,103.53)
STD=(58.395,57.12,57.375)
PRODUCTION_GEOMETRY={'deepglobe':(2448,7),'inria':(5000,2)}


def source_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def scale_size(side):
    """Input side length using Python round at the configured scale."""
    return max(1,round(side*SCALE))


class ISDNetUnifiedS08:
    def __init__(self,model,dataset,precision,config,*,device='cuda',cpu_synthetic=False):
        if dataset not in PRODUCTION_GEOMETRY or precision not in ('fp32','fp16'):
            raise ValueError('Dataset/precision outside the supported settings')
        self.device=torch.device(device)
        self.cpu_synthetic=bool(cpu_synthetic)
        if self.device.type=='cpu':
            if not self.cpu_synthetic or precision!='fp32':
                raise ValueError('CPU route execution is restricted to explicitly synthetic FP32 QA')
        elif self.device.type!='cuda' or self.device.index not in (None,0) or self.cpu_synthetic:
            raise ValueError('Formal route uses the single default CUDA device')
        self.model,self.dataset,self.precision=model,dataset,precision
        data=config['datasets'][dataset]
        self.side,self.classes=data['side'],data['classes']
        if self.classes!=PRODUCTION_GEOMETRY[dataset][1]:
            raise ValueError('Configured class vocabulary differs')
        if not self.cpu_synthetic and self.side!=PRODUCTION_GEOMETRY[dataset][0]:
            raise ValueError('Configured production geometry differs')
        if not self.cpu_synthetic and scale_size(self.side)!=int(self.side*SCALE):
            raise ValueError('Generic-round and original official shapes must agree on configured images')
        if getattr(model,'align_corners',False) is not False:
            raise ValueError('Original official model uses align_corners=False')
        test_cfg=getattr(model,'test_cfg',None)
        mode=getattr(test_cfg,'mode',None) if not isinstance(test_cfg,dict) else test_cfg.get('mode')
        if mode!='whole':
            raise ValueError('Original official ISDNet whole-mode configuration required')
        for parameter in model.parameters():
            if parameter.device.type!=self.device.type or (self.device.index is not None and parameter.device.index!=self.device.index):
                raise ValueError('Model and endpoint devices differ')
        self.mean=torch.tensor(MEAN,device=self.device,dtype=torch.float32)[None,:,None,None]
        self.std=torch.tensor(STD,device=self.device,dtype=torch.float32)[None,:,None,None]

    def preprocess(self,rgb):
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.shape!=(self.side,self.side,3):
            raise ValueError('Endpoint starts with decoded fixed-size CPU RGB uint8')
        # Transfer the original uint8 image to the execution device first. Float
        # bilinear interpolation uses the original generic R50/B5 size/mode/
        # align_corners policy. The GPU-path order is resize then normalize.
        batch=torch.as_tensor(rgb.copy(),device=self.device).permute(2,0,1).float()[None]
        target=scale_size(self.side)
        scaled=F.interpolate(batch,size=(target,target),mode='bilinear',align_corners=False)
        normalized=(scaled-self.mean)/self.std
        # The original official endpoint has no external padding/divisor resize.
        meta=dict(ori_shape=(self.side,self.side,3),img_shape=(target,target,3),
                  pad_shape=(target,target,3),flip=False)
        return normalized,[meta]

    @torch.no_grad()
    def __call__(self,rgb):
        batch,meta=self.preprocess(rgb)
        autocast=torch.autocast('cuda',dtype=torch.float16,enabled=True) if self.precision=='fp16' else contextlib.nullcontext()
        with autocast:
            logits=self.model.encode_decode(batch,meta)
        if logits.shape[:2]!=(1,self.classes) or tuple(logits.shape[2:])!=tuple(batch.shape[2:]):
            raise ValueError('Official encode_decode must return input-size logits with the configured vocabulary')
        # Existing ISDNet internals, including floor downsampling and reflection
        # padding, remain in the unchanged model. No extra padding is introduced.
        canvas=F.interpolate(logits.float(),size=(self.side,self.side),mode='bilinear',align_corners=False)[0].contiguous()
        return canvas.argmax(0)


def make_isdnet_unified_s08(model,dataset,precision,config,*,device='cuda',cpu_synthetic=False):
    """Return one callable suitable for TimingHarness or E1.measure_path."""
    return ISDNetUnifiedS08(model,dataset,precision,config,device=device,cpu_synthetic=cpu_synthetic)


def add_isdnet_s08_call(model,original_calls,dataset,precision,config,*,device='cuda',cpu_synthetic=False):
    """Return a new dictionary preserving both original call objects unchanged."""
    if list(original_calls)!=['ISDNet_unified','ISDNet_official_down_0.8']:
        raise ValueError('Expected the exact two unchanged ISDNet routes in original order')
    calls=dict(original_calls)
    calls[ROUTE_NAME]=make_isdnet_unified_s08(model,dataset,precision,config,device=device,cpu_synthetic=cpu_synthetic)
    return calls


