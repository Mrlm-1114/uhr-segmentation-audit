"""Measured DG/Inria inference arithmetic, with portable construction."""
import math
import numpy as np
import torch
import torch.nn.functional as F

class InferencePaths:
    def __init__(self, model, dataset, family, precision, config):
        self.model, self.family, self.precision = model, family, precision
        data = config['datasets'][dataset]
        self.side, self.classes = data['side'], data['classes']
        self.window, self.stride = config['slide']['window'], config['slide']['stride']
        self.mean = torch.tensor([123.675, 116.28, 103.53], device='cuda')[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375], device='cuda')[:, None, None]
        self.scales = config['scales']

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.float16, enabled=self.precision == 'fp16')

    def segment(self, batch):
        h, w = batch.shape[-2:]
        meta = dict(ori_shape=(h, w, 3), img_shape=(h, w, 3), pad_shape=(h, w, 3), flip=False)
        with self.autocast():
            return self.model.encode_decode(batch, [meta]).float()

    def normalize(self, rgb):
        return (torch.as_tensor(rgb.copy(), device='cuda').permute(2, 0, 1).float() - self.mean) / self.std

    def scaled_logits(self, normalized, scale, mode):
        size = max(1, round(self.side * scale))
        x = F.interpolate(normalized[None], size=(size, size), mode='bilinear', align_corners=False)
        if mode == 'whole':
            return self.segment(x)[0]
        if mode != 'slide' or scale != 1:
            raise ValueError('Only the separately named native slide path is permitted')
        out = torch.zeros((1, self.classes, size, size), device='cuda')
        count = torch.zeros((1, 1, size, size), device='cuda')
        steps = max(math.ceil((size - self.window) / self.stride) + 1, 1)
        for row in range(steps):
            y1 = min(row * self.stride, max(size - self.window, 0)); y2 = min(y1 + self.window, size)
            for col in range(steps):
                x1 = min(col * self.stride, max(size - self.window, 0)); x2 = min(x1 + self.window, size)
                out[:, :, y1:y2, x1:x2] += self.segment(x[:, :, y1:y2, x1:x2])
                count[:, :, y1:y2, x1:x2] += 1
        return (out / count)[0]

    @torch.no_grad()
    def unified(self, rgb, scale=1., mode='whole'):
        x = self.normalize(rgb)
        if self.family == 'isdnet':
            # Exactly the unified reference.py endpoint; no redundant interpolate.
            return self.segment(x[None])[0].argmax(0)
        logits = self.scaled_logits(x, scale, mode)
        canvas = F.interpolate(logits[None].float(), size=(self.side, self.side),
                               mode='bilinear', align_corners=False)[0].contiguous()
        return canvas.argmax(0)

    @torch.no_grad()
    def official_default(self, rgb):
        if self.family != 'isdnet':
            raise ValueError('Official default path is ISDNet only')
        import mmcv
        target = int(self.side * .8)
        image = mmcv.imrescale(rgb, (target, target), interpolation='bilinear', backend='cv2')
        image = mmcv.imnormalize(image, np.array([123.675, 116.28, 103.53], dtype=np.float32),
                                np.array([58.395, 57.12, 57.375], dtype=np.float32), to_rgb=False)
        batch = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to('cuda')
        meta = dict(ori_shape=(self.side, self.side, 3), img_shape=(*image.shape[:2], 3),
                    pad_shape=(*image.shape[:2], 3), flip=False)
        with self.autocast():
            # Includes the official rescale -> softmax -> argmax order.
            probabilities = self.model.inference(batch, [meta], rescale=True)
            return probabilities.argmax(dim=1)[0]

    def calls(self):
        import functools
        if self.family == 'isdnet':
            return {'ISDNet_unified': self.unified, 'ISDNet_official_down_0.8': self.official_default}
        calls = {f'S_{s:g}': functools.partial(self.unified, scale=s) for s in self.scales}
        calls['native_slide_1024_stride_768'] = functools.partial(self.unified, scale=1., mode='slide')
        return calls
