"""Evaluate or time locally supplied models and local input manifests."""
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--isdnet-repo',type=Path);p.add_argument('--route',action='append')
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp32');p.add_argument('--time',action='store_true')
    p.add_argument('--label-encoding',choices=['indexed','deepglobe_rgb','inria_binary'],default='indexed')
    args=p.parse_args()
    from uhr_eval.timing import configure,measure
    configure()
    from uhr_eval.models import calls
    from uhr_eval.io import inputs,image,label
    from uhr_eval.metrics import confusion,summarize
    import numpy as np,torch
    cfg=json.loads(args.config.read_text(encoding='utf-8'));model,routes=calls(cfg,args.checkpoint,args.isdnet_repo,args.precision)
    if args.route:routes={name:routes[name] for name in args.route}
    rows=inputs(args.manifest);ds=cfg['dataset'];classes={'deepglobe':7,'inria':2,'urur':8}[ds]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.time:
        decoded=[image(x['image']) for x in rows]
        with args.output.open('x',encoding='utf-8',newline='') as f:
            writer=None
            for row in measure(routes,decoded):
                if writer is None:writer=csv.DictWriter(f,list(row));writer.writeheader()
                writer.writerow(row);f.flush()
    else:
        results=[]
        for name,call in routes.items():
            total=np.zeros((classes,classes),dtype=np.int64);per_image=[]
            try:
                for index,row in enumerate(rows):
                    rgb=image(row['image']);gt=label(row['label'],ds,args.label_encoding)
                    with torch.no_grad():pred=call(rgb).cpu().numpy()
                    cm=confusion(pred,gt,classes);total+=cm
                    if ds!='urur':per_image.append(dict(index=index,**summarize(cm,ds)))
                results.append(dict(route=name,status='complete',n_images=len(rows),**summarize(total,ds),per_image=per_image))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();results.append(dict(route=name,status='unavailable',reason='out_of_memory'))
        with args.output.open('x',encoding='utf-8') as f:json.dump({'dataset':ds,'precision':args.precision,'routes':results},f,indent=2)

if __name__=='__main__':main()
