"""Train the supplied recipe locally; input labels must use the documented indices."""
import argparse,copy,json,random
from contextlib import nullcontext
from functools import partial
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--split',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--isdnet-repo',type=Path)
    p.add_argument('--shallow-weight',type=Path);args=p.parse_args()
    import numpy as np,torch
    from uhr_eval.runtime import prepare_mmcv
    prepare_mmcv()
    from mmcv import Config
    from mmcv.parallel import collate
    from mmseg.core import build_optimizer
    from mmseg.datasets import build_dataset
    from uhr_eval.models import build
    from uhr_eval.sampling import SeededDataset,TrainingBatches,initialize_worker
    cfg=json.loads(args.config.read_text(encoding='utf-8'));seed=cfg['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=cfg['cudnn_benchmark']
    cfg['data']['train'].update(data_root=str(args.data_root),split=str(args.split))
    for stage in cfg['data']['train']['pipeline']:
        for key in ['img_scale','crop_size','ratio_range','size','mean','std','degree']:
            if isinstance(stage.get(key),list):stage[key]=tuple(stage[key])
    model=build(cfg,isdnet_repo=args.isdnet_repo,training=True,shallow_weight=args.shallow_weight)
    optimizer=build_optimizer(model,Config({'optimizer':cfg['optimizer']}).optimizer);base=[x['lr'] for x in optimizer.param_groups]
    dataset=build_dataset(Config({'train':cfg['data']['train']}).train)
    batch=cfg.get('micro_batch',cfg['data']['samples_per_gpu'])*cfg.get('accumulation_steps',1)
    steps=cfg['runner']['max_iters'];workers=cfg['data']['workers_per_gpu']
    loader=torch.utils.data.DataLoader(SeededDataset(dataset,seed),batch_sampler=TrainingBatches(len(dataset),batch,steps,0,seed),
        num_workers=workers,persistent_workers=bool(workers),pin_memory=True,worker_init_fn=initialize_worker,
        generator=torch.Generator().manual_seed(seed),collate_fn=partial(collate,samples_per_gpu=batch))
    scaler=torch.cuda.amp.GradScaler() if cfg['precision']=='fp16' else None
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'recipe.json').open('w',encoding='utf-8') as f:json.dump(cfg,f,indent=2)
    for iteration,data in enumerate(loader):
        rate=cfg.get('lr_config',{});power=rate.get('power',.9);minimum=rate.get('min_lr',0)
        for group,lr in zip(optimizer.param_groups,base):
            group['lr']=(lr-minimum)*(1-iteration/steps)**power+minimum
            if cfg['family']=='B5' and iteration<1500:group['lr']*=1e-6+(1-1e-6)*iteration/1500
        optimizer.zero_grad(set_to_none=True)
        x=data['img'].data[0];y=data['gt_semantic_seg'].data[0];meta=data['img_metas'].data[0]
        micro=cfg.get('micro_batch',batch);accum=cfg.get('accumulation_steps',1)
        if len(x)!=micro*accum:raise ValueError('Microbatch does not match effective batch')
        for start in range(0,len(x),micro):
            with torch.cuda.amp.autocast() if scaler else nullcontext():
                loss,_=model._parse_losses(model.forward_train(x[start:start+micro].cuda(non_blocking=True),meta[start:start+micro],y[start:start+micro].cuda(non_blocking=True)))
                loss=loss/accum
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            if scaler:scaler.scale(loss).backward()
            else:loss.backward()
        if scaler:scaler.step(optimizer);scaler.update()
        else:optimizer.step()
        if (iteration+1)%2000==0 or iteration+1==steps:
            torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'iteration':iteration+1,'dataset':cfg['dataset'],'seed':seed},args.output/f'iter_{iteration+1}.pth')
        if (iteration+1)%50==0:print(iteration+1,float(loss.detach()),flush=True)

if __name__=='__main__':main()
