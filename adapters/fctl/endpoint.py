# Copyright (c) 2021 liqi (FCtL upstream). MIT; see LICENSE.
# Adapter copyright (c) 2026 Keyao Ma.
"""MIT FCtL endpoint: unchanged official models and Evaluator, explicit device IO."""
import hashlib
import json
from pathlib import Path
import sys

class FCtLEndpoint:
    def __init__(self, code, weights, receipt, device='cuda', patch=508, sub_batch=16):
        import torch, torchvision
        self.torch=torch;self.device=torch.device(device)
        self.code=Path(code).resolve();self.weights=Path(weights).resolve()
        receipt=json.loads(Path(receipt).read_text())
        for row in receipt['source_files']:
            assert hashlib.sha256((self.code/row['path']).read_bytes()).hexdigest()==row['sha256']
        sys.path.insert(0,str(self.code))
        import helper
        self.helper=helper
        vgg=torchvision.models.vgg16;loader=torch.load;load_state=torch.nn.Module.load_state_dict
        pending={};coverage=[]
        def offline_vgg(pretrained=True,*args,**kwargs):
            assert pretrained is True and not args and not kwargs
            return vgg(weights=None,init_weights=False)
        def verified_checkpoint(path,*args,**kwargs):
            path=Path(path).resolve()
            assert path.parent==self.weights and path.name in {r['name'] for r in receipt['files']}
            assert not pending
            checkpoint=loader(path,map_location='cpu',weights_only=True)
            pending.update(name=path.name,state=checkpoint)
            return checkpoint
        def complete_load(module,state,*args,**kwargs):
            assert pending
            own=module.state_dict();checkpoint=pending['state']
            assert set(own)<=set(checkpoint),'Incomplete checkpoint'
            result=load_state(module,state,*args,**kwargs)
            assert not result.missing_keys and not result.unexpected_keys
            # Exact tensor coverage was checked on CPU. GPU copies retain dtype.
            for key,value in module.state_dict().items():
                assert torch.equal(value.cpu(),checkpoint[key]),key
            coverage.append(dict(name=pending['name'],state_keys=len(own),all_final_tensors_equal=True))
            pending.clear()
            return result
        torchvision.models.vgg16=offline_vgg;torch.load=verified_checkpoint
        torch.nn.Module.load_state_dict=complete_load
        try:
            self.models=helper.create_model_load_weights(2,str(self.weights/'all.epoch.pth'),
                str(self.weights/'B10.epoch.pth'),str(self.weights/'B15.epoch.pth'),mode=3)
        finally:
            torchvision.models.vgg16=vgg;torch.load=loader;torch.nn.Module.load_state_dict=load_state
        assert len(coverage)==3 and not pending
        self.coverage=coverage
        for model in self.models:model.eval()
        self.evaluators={name:helper.Evaluator(2,(patch,patch),(patch,patch),sub_batch_size=sub_batch,
                            mode=3,val=no_tta,dataset=2,context10=2,context15=3)
                         for name,no_tta in [('FCtL',False),('FCtL-noTTA',True)]}

    def to_endpoint(self, prediction):
        return self.torch.as_tensor(prediction,device=self.device)

    def predict(self,rgb,variant='FCtL'):
        from PIL import Image
        sample={'image':[Image.fromarray(rgb)],'id':['decoded_input'],'label':[]}
        with self.torch.no_grad():
            predictions=self.evaluators[variant].eval_test(sample,*self.models)
        assert len(predictions)==1 and predictions[0].shape==rgb.shape[:2]
        # Official CPU overlap/TTA accumulation is retained; final H2D is timed.
        return self.to_endpoint(predictions[0])
