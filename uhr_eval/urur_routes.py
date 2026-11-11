"""Measured native-scale URUR endpoints."""
import numpy as np
import torch
from .routes import InferencePaths
SCALES=[1/8,1/6,1/4,1/3,1/2,3/4,1]
ROUTES=["scale_1_8","scale_1_6","scale_1_4","scale_1_3","scale_1_2","scale_3_4","scale_1","slide_1024_stride_768"]

class URURPaths(InferencePaths):
    def __init__(self, model, family, side=5120, device='cuda', window=1024, stride=768):
        # Same generic arithmetic; device parameter only permits synthetic CPU QA.
        self.model = model; self.family = 'isdnet' if family == 'ISDNet' else family
        self.precision = 'fp32'; self.side = side; self.classes = 8
        self.window = window; self.stride = stride; self.scales = SCALES; self.device = device
        self.mean = torch.tensor([123.675,116.28,103.53], device=device)[:,None,None]
        self.std = torch.tensor([58.395,57.12,57.375], device=device)[:,None,None]

    def normalize(self, rgb):
        return (torch.as_tensor(rgb.copy(), device=self.device).permute(2,0,1).float()-self.mean)/self.std

    @torch.no_grad()
    def official_default(self, rgb):
        if self.family != 'isdnet':
            raise ValueError('Official path is ISDNet only')
        batch, meta = self.official_cpu(rgb)
        return self.official_gpu((batch, meta))

    def official_cpu(self, rgb):
        import mmcv
        # Official URUR nominal scale is 1, not the DG/Inria scale 0.8.
        image = mmcv.imrescale(rgb, (self.side,self.side), interpolation='bilinear', backend='cv2')
        image = mmcv.imnormalize(image, np.array([123.675,116.28,103.53], dtype=np.float32),
                                np.array([58.395,57.12,57.375], dtype=np.float32), to_rgb=False)
        batch = torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1)))[None]
        shape = (*image.shape[:2],3)
        meta = dict(ori_shape=(self.side,self.side,3),img_shape=shape,pad_shape=shape,flip=False)
        return batch, meta

    @torch.no_grad()
    def official_gpu(self, prepared):
        batch, meta = prepared
        probabilities = self.model.inference(batch.to(self.device), [meta], rescale=True)
        return probabilities.argmax(1)[0]

    def route_calls(self):
        from functools import partial
        if self.family == 'isdnet':
            return {'ISDNet_official':self.official_default, 'ISDNet_gpu':self.unified}
        out = {name:partial(self.unified,scale=scale) for name,scale in zip(ROUTES[:7],SCALES)}
        out[ROUTES[-1]] = partial(self.unified,scale=1.,mode='slide')
        return out
