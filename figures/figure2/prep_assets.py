"""Rebuild the raster assets of fig_framework.tex (Fig. 2).

Usage (run with an isolated interpreter):
    python3 -I prep_assets.py <thumb_dir> <qual_dir> <out_dir>

Inputs (not shipped here; the caller supplies the authorized crops):
    <thumb_dir>/deepglobe_crop.jpg        DeepGlobe crop      (externally supplied crop)
    <thumb_dir>/inria_crop.jpg    Inria crop          (externally supplied crop)
    <thumb_dir>/urur_crop.jpg   URUR crop           (externally supplied crop)
    <qual_dir>/in_rgb.png              Inria austin7 RGB   (Fig. 4 panel)
    <qual_dir>/in_b5.png               SegFormer-B5 prediction for the same crop (Fig. 4 panel)

Outputs (420 x 420 px JPEG, quality 92):
    ds_deepglobe.jpg        plain crop, used for the seven-scale glyph
    ds_*_tag.jpg            dataset thumbnails with a darkened bottom strip for the size label
    route_sliding.jpg       Inria crop dimmed except for the active sliding window
    path_input.jpg          input endpoint of the timed path
    path_output.jpg         predicted buildings tinted on the same crop (output endpoint)

Darkening is baked into the pixels so that the figure PDF contains no transparency.
Requires Pillow and NumPy.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

thumb_dir, qual_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = 420
INK = np.array([23, 32, 42], np.float32)          # figure text colour #17202A


def save(im, name):
    im.convert('RGB').resize((N, N), Image.LANCZOS).save(
        f'{out}/{name}', quality=92, optimize=True, subsampling=0)


def save_arr(a, name):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(
        f'{out}/{name}', quality=92, optimize=True, subsampling=0)


# ---- dataset thumbnails ----
save(Image.open(f'{thumb_dir}/deepglobe_crop.jpg'), 'ds_deepglobe.jpg')
save(Image.open(f'{thumb_dir}/inria_crop.jpg'), 'ds_inria.jpg')
save(Image.open(f'{thumb_dir}/urur_crop.jpg'), 'ds_urur.jpg')

# ---- endpoints of the timed path: Inria austin7 RGB and the SegFormer-B5 prediction ----
rgb = Image.open(f'{qual_dir}/in_rgb.png').convert('RGB')
msk = Image.open(f'{qual_dir}/in_b5.png').convert('L')
save(rgb, 'path_input.jpg')

a = np.asarray(rgb).astype(np.float32)
m = np.asarray(msk) > 127                           # predicted building pixels
tint = np.array([38, 172, 205], np.float32)
o = a * 0.55 + 18                                   # background, slightly dimmed
o[m] = a[m] * 0.38 + tint * 0.62
mi = Image.fromarray((m * 255).astype(np.uint8))    # thin outline around predicted buildings
edge = (np.asarray(mi.filter(ImageFilter.MaxFilter(5))).astype(np.int16)
        - np.asarray(mi.filter(ImageFilter.MinFilter(5))).astype(np.int16)) > 0
o[edge] = o[edge] * 0.25 + np.array([10, 70, 95], np.float32) * 0.75
save(Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)), 'path_output.jpg')

# ---- size-label strip on the dataset thumbnails (bottom 30.5 %, 74 % ink) ----
for n in ('deepglobe', 'inria', 'urur'):
    im = np.asarray(Image.open(f'{out}/ds_{n}.jpg').convert('RGB')).astype(np.float32)
    y0 = int(round(im.shape[0] * (1 - 0.305)))
    im[y0:] = im[y0:] * (1 - 0.74) + INK * 0.74
    save_arr(im, f'ds_{n}_tag.jpg')

# ---- sliding-window glyph: dim everything except the active window ----
# window given as fractions of the side, origin at the lower-left corner (matches the TikZ outlines)
im = np.asarray(Image.open(f'{out}/ds_inria.jpg').convert('RGB')).astype(np.float32)
h, w = im.shape[:2]
dim = im * 0.50 + INK * 0.50
wx0, wy0, ws = 0.0862, 0.4914, 0.4224
x0, x1 = int(round(wx0 * w)), int(round((wx0 + ws) * w))
y1, y0 = h - int(round(wy0 * h)), h - int(round((wy0 + ws) * h))
dim[y0:y1, x0:x1] = im[y0:y1, x0:x1]
save_arr(dim, 'route_sliding.jpg')

print('assets written to', out)
