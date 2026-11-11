"""Portable construction around the pinned MMSegmentation and ISDNet code."""
from pathlib import Path
import copy, importlib.util, json, sys


def register_isdnet(repo,shallow_weight=None):
    import mmseg.models.decode_heads.stdc_head as modern
    base=Path(repo)/'mmseg/models'
    def load(name,path):
        if name in sys.modules:return sys.modules[name]
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
        sys.modules[name]=m;spec.loader.exec_module(m);return m
    shallow=load('mmseg.models.decode_heads.public_isd_stdc',base/'decode_heads/stdc_head.py')
    class LocalShallow(shallow.ShallowNet):
        def __init__(self,*a,**kw):
            kw['pretrain_model']=str(shallow_weight) if shallow_weight else ''
            super().__init__(*a,**kw)
    modern.ShallowNet=LocalShallow
    for name in ['refine_decode_head','aspp_refine_head','isdhead']:
        m=load('mmseg.models.decode_heads.'+name,base/f'decode_heads/{name}.py')
        if name=='isdhead':m.ShallowNet=LocalShallow
    load('mmseg.models.segmentors.encoder_decoder_refine',base/'segmentors/encoder_decoder_refine.py')


def build(cfg,checkpoint=None,isdnet_repo=None,training=False,shallow_weight=None):
    import torch
    from .runtime import prepare_mmcv
    prepare_mmcv()
    from mmcv import Config
    from mmcv.cnn.utils import revert_sync_batchnorm
    from mmseg.models import build_segmentor
    model_cfg=copy.deepcopy(cfg['model'])
    if cfg['family']=='ISDNet':register_isdnet(isdnet_repo,shallow_weight if training else None)
    if not training:
        model_cfg['pretrained']=None
        model_cfg['backbone']['init_cfg']=None
        model_cfg['backbone']['pretrained']=None
    model=build_segmentor(Config({'model':model_cfg}).model)
    if training:model.init_weights()
    model=revert_sync_batchnorm(model)
    if checkpoint:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state={k.removeprefix('module.'):v for k,v in saved.get('state_dict',saved).items()}
        if cfg['family']=='ISDNet' and not any(k.startswith('refine_head.sr_decoder.') for k in state):
            if hasattr(model.refine_head,'sr_decoder'):del model.refine_head.sr_decoder
        model.load_state_dict(state,strict=True)
    model.out_channels=8 if cfg['dataset']=='urur' else (7 if cfg['dataset']=='deepglobe' else 2)
    return model.cuda().train(training)


def calls(cfg,checkpoint,isdnet_repo=None,precision='fp32'):
    from .routes import InferencePaths
    model=build(cfg,checkpoint,isdnet_repo)
    ds=cfg['dataset'];family=cfg['family']
    if ds=='urur':
        from .urur_routes import URURPaths
        return model,URURPaths(model,family).route_calls()
    spec={'datasets':{ds:{'side':2448 if ds=='deepglobe' else 5000,'classes':7 if ds=='deepglobe' else 2}},
          'slide':{'window':1024,'stride':768},'scales':[1/8,1/6,1/4,1/3,1/2,3/4,1]}
    family={'R50':'dlv3','B5':'segformer_b5','ISDNet':'isdnet'}[family]
    endpoint=InferencePaths(model,ds,family,precision,spec);routes=endpoint.calls()
    if family=='isdnet':
        from adapters.isdnet_gpu import add_isdnet_s08_call
        routes=add_isdnet_s08_call(model,routes,ds,precision,spec)
    return model,routes
