# Third-party software

The repository's MIT license covers original study code by Keyao Ma. It does not
replace upstream licenses. The table records the source revisions associated
with the evaluated implementations. Package dependencies are installed externally.

| Component | Source and pinned commit | License | Distribution here |
|---|---|---|---|
| PyTorch 2.0.0+cu118 | [PyTorch](https://github.com/pytorch/pytorch/tree/c263bd43e8e8502d4726643bc6fd046f0130ac0e) | BSD-3-Clause, third-party notices apply | Dependency only |
| MMCV 1.7.2 | [MMCV](https://github.com/open-mmlab/mmcv/tree/4c01b026f0afa5a91a5f54aea313788da1e40f95) | Apache-2.0; per-file exceptions apply | Dependency only |
| MMSegmentation 0.30.0 | [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/ed839828760a5f6193822e0bf3492b88ae6140da) | Apache-2.0 with LICENSES.md exceptions | Dependency only; no library source copied |
| SegFormer implementation within MMSegmentation | [Pinned LICENSES.md](https://github.com/open-mmlab/mmsegmentation/blob/ed839828760a5f6193822e0bf3492b88ae6140da/LICENSES.md) | NVIDIA Source Code License for designated SegFormer code; research/evaluation restrictions apply | Dependency only |
| ISDNet | [ISDNet](https://github.com/cedricgsh/ISDNet/tree/7341335677fd6201457b321e1e6368feb7ad647e) | [Apache-2.0](https://github.com/cedricgsh/ISDNet/blob/7341335677fd6201457b321e1e6368feb7ad647e/LICENSE), Copyright 2020 The MMSegmentation Authors | External checkout; study adapters only |
| FCtL | [FCtL](https://github.com/liqiokkk/FCtL/tree/6fdb0c02f22efbf267781857c9ee7f7ebc951974) | MIT, Copyright (c) 2021 liqi | Adapter and original license in `adapters/fctl/`; external model code |
| MagNet / MagNet-Fast | [MagNet](https://github.com/VinAIResearch/MagNet/tree/7f5b8760cb471230402015d80117559de0e38600) | AGPL-3.0 | Adapter and full AGPL text in `adapters/magnet_agpl/`; external model code |

Placing AGPL files in a directory documents their scope; it does not remove AGPL
obligations for combined or derivative works. Follow the upstream terms when
using or distributing that integration. No dataset, model weight, upstream
repository history, or publication PDF is included.

The exact installed package versions are recorded in `environment/`. NumPy,
Pillow, OpenCV, Matplotlib, torchvision and other installed packages retain their
own licenses. The main environment freeze lists their distribution versions;
source commits were not recorded for every transitive binary distribution.
