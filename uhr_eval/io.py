"""Local input manifests and aggregate URUR output records."""
import json
from pathlib import Path

def inputs(manifest):
    path=Path(manifest);rows=json.loads(path.read_text(encoding='utf-8'))
    return [dict(image=path.parent/r['image'],label=path.parent/r['label']) for r in rows]

def image(path):
    import mmcv
    return mmcv.imread(str(path),channel_order='rgb')

def label(path,dataset,encoding='indexed'):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:a=np.array(im)
    if encoding=='deepglobe_rgb':
        if not np.all((a==0)|(a==255)):raise ValueError('Unexpected palette')
        code=(a[...,0]//255)*4+(a[...,1]//255)*2+a[...,2]//255
        a=np.array([0,5,4,1,255,3,2,6],dtype=np.uint8)[code]
        if np.any(a==255):raise ValueError('Unexpected palette color')
    elif encoding=='inria_binary':a=(a//255).astype(np.uint8)
    classes={'deepglobe':7,'inria':2,'urur':8}[dataset]
    if a.ndim!=2 or np.any((a>=classes)&(a!=255)):raise ValueError('Unexpected label vocabulary')
    return a
