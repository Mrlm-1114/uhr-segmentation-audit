"""Resolve a pinned external ISDNet config into a portable study JSON."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--dataset',choices=['deepglobe','inria'],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from mmcv import Config
    name={'deepglobe':'isdnet_r18-d8_1224x1224_80k_DeepGlobe.py','inria':'isdnet_r18-d8_2500x2500_40k_InriaAerial.py'}[args.dataset]
    cfg=Config.fromfile(str(args.repo/'configs/isdnet'/name)).to_dict()
    model=cfg['model'];model['pretrained']=None
    for component in [model['backbone'],model['auxiliary_head'],*model['decode_head']]:component['norm_cfg']={'type':'BN','requires_grad':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump({'model':model,'dataset':args.dataset,'family':'ISDNet'},f,indent=2)

if __name__=='__main__':main()
