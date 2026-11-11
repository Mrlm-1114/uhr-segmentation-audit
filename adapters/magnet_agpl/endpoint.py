# SPDX-License-Identifier: AGPL-3.0-only
"""AGPL-3.0 complete-path adapter using the unchanged official inference loop.

The loop is selected from the SHA-pinned test.py AST. No arithmetic, sorting,
selection or refinement expression is rewritten. Disk decode and metrics stay
outside the endpoint; official CPU patch preparation remains inside it.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

class MagNetEndpoint:
    def __init__(self, code, weights, receipt, device='cuda',
                 scales=((612,612),(1224,1224),(2448,2448)), crop=(612,612), network=(508,508)):
        import numpy as np
        import torch
        import torch.utils.model_zoo as model_zoo
        self.torch = torch
        self.device = torch.device(device)
        self.code = Path(code).resolve()
        self.weights = Path(weights).resolve()
        receipt = json.loads(Path(receipt).read_text('utf-8'))
        row = next(r for r in receipt['source_files'] if r['path']=='test.py')
        source = (self.code/'test.py').read_bytes()
        assert hashlib.sha256(source).hexdigest() == row['sha256']
        self.source_sha256 = row['sha256']
        sys.path.insert(0, str(self.code))
        np.float=float; np.bool=bool; np.int=int
        spec = importlib.util.spec_from_file_location('magnet_pinned_official', self.code/'test.py')
        official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
        self.official = official
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        image_loop = next(n for n in main.body if isinstance(n,ast.For) and
                          isinstance(n.iter,ast.Call) and ast.unparse(n.iter)=='enumerate(dataloader)')
        scale_loop = next(n for n in image_loop.body if isinstance(n,ast.For) and
                          ast.unparse(n.iter)=='enumerate(zip(patch_coords, opt.scales))')
        fn = ast.parse('def official_core(model, refinement_models, median_blur, patch_coords, opt, image_patches, scale_idx, device):\n    sub_batch_size=opt.sub_batch_size\n    final_output=None\n    coarse_pred=None\n').body[0]
        fn.body += [copy.deepcopy(scale_loop), ast.parse('return final_output, coarse_pred').body[0]]
        module = ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        namespace = dict(official.__dict__)
        exec(compile(module, str(self.code/'test.py')+'::unchanged_scale_loop', 'exec'), namespace)
        self.core = namespace['official_core']
        self.loop_ast_sha256 = hashlib.sha256(ast.dump(scale_loop,include_attributes=False).encode()).hexdigest()
        self.opt = SimpleNamespace(dataset='deepglobe',phase='test',root='',datalist='',
                                   scales=list(scales),crop_size=crop,input_size=network,num_classes=7,
                                   smooth_kernel=11,n_points=.75,n_patches=-1,sub_batch_size=1)
        self.dataset = official.get_dataset_with_name('deepglobe')(self.opt)
        state = torch.load(self.weights/'deepglobe_fpn.pth',map_location='cpu',weights_only=True)
        backbone = {k.removeprefix('model.resnet_backbone.'):v for k,v in state.items()
                    if k.startswith('model.resnet_backbone.')}
        previous = model_zoo.load_url
        def offline_init(url,*args,**kwargs):
            assert url=='https://download.pytorch.org/models/resnet50-19c8e357.pth'
            return backbone
        model_zoo.load_url = offline_init
        try: self.model = official.get_model_with_name('fpn')(7)
        finally: model_zoo.load_url = previous
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.refiner = official.RefinementMagNet(7,use_bn=True)
        ref_state = torch.load(self.weights/'deepglobe_refinement.pth',map_location='cpu',weights_only=True)
        compatible = self.refiner.load_state_dict(ref_state,strict=False)
        assert compatible.missing_keys==[] and compatible.unexpected_keys==['error_conv.weight']
        self.refiner.to(self.device).eval()
        self.coords = [torch.tensor(official.get_patch_coords(scale,crop)).to(self.device) for scale in scales]
        self.blur = official.MedianBlur(kernel_size=(11,11)).to(self.device).eval()

    def prepare(self, rgb):
        # The pinned test.py keeps image_patches and scale_idx on CPU while
        # selected_patch_ids comes from a CUDA sort.  Current PyTorch rejects
        # that mixed-device advanced index.  Moving the two data tensors
        # together preserves the official values and indexing order while
        # keeping the compatibility fix outside the AGPL source tree.
        image_patches, scale_idx = self.dataset.slice_image(rgb)
        return image_patches.to(self.device), scale_idx.to(self.device)

    def compute(self, prepared, variant='MagNet', include_coarse=True):
        opt = copy.copy(self.opt)
        if variant=='MagNet-Fast':
            opt.n_points=.9; opt.n_patches=3; opt.sub_batch_size=2
        elif variant=='Baseline':
            opt.n_patches=0
        else: assert variant=='MagNet'
        with self.torch.no_grad():
            final, coarse = self.core(self.model,[self.refiner],self.blur,self.coords,opt,
                                      prepared[0],prepared[1],self.device)
            side = opt.scales[-1][::-1]
            prediction = self.official.F.interpolate(final,side,mode='bilinear',align_corners=False).argmax(1)[0]
            coarse_prediction = (self.official.F.interpolate(coarse,side,mode='bilinear',align_corners=False).argmax(1)[0]
                                 if include_coarse else None)
            return prediction, coarse_prediction

    def predict(self, rgb, variant='MagNet'):
        final, coarse = self.compute(self.prepare(rgb), variant, include_coarse=False)
        del coarse
        return final
