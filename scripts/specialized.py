"""Run a separately licensed endpoint through its public predict interface."""
import argparse,csv,importlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factory',required=True,help='module:class');p.add_argument('--code',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--dataset',choices=['deepglobe','inria'],required=True)
    p.add_argument('--variant',required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--time',action='store_true')
    p.add_argument('--label-encoding',default='indexed',choices=['indexed','deepglobe_rgb','inria_binary'])
    args=p.parse_args()
    from uhr_eval.timing import configure,measure
    configure()
    from uhr_eval.io import inputs,image,label
    from uhr_eval.metrics import confusion,summarize
    import numpy as np
    module,cls=args.factory.split(':');endpoint=getattr(importlib.import_module(module),cls)(args.code,args.weights,args.receipt)
    call=lambda rgb:endpoint.predict(rgb,args.variant)
    rows=inputs(args.manifest);args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.time:
        decoded=[image(x['image']) for x in rows];data=list(measure({args.variant:call},decoded))
        with args.output.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    else:
        count=7 if args.dataset=='deepglobe' else 2;cm=np.zeros((count,count),dtype=np.int64)
        for x in rows:cm+=confusion(call(image(x['image'])).cpu().numpy(),label(x['label'],args.dataset,args.label_encoding),count)
        with args.output.open('x',encoding='utf-8') as f:json.dump(summarize(cm,args.dataset),f,indent=2)

if __name__=='__main__':main()
