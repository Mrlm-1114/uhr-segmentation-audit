# Configuration provenance

DeepGlobe and Inria B5 configurations cover measured seeds 0, 1 and 2. Their
training schedules and optimizers retain the executed configurations.
The measured R50 configuration uses seed 20260929 (`r50_historical.json`).

URUR configurations come from the executed final training recipes: R18, R50 and
B5 use 57970 optimizer updates; ISDNet uses 160000 updates, microbatch 2 and
four-step accumulation (effective batch 8). Data paths and initializer paths
are portable references. Supply authorized URUR splits outside this repository.

The training reader expects indexed labels: DeepGlobe 0 unknown, 1 urban,
2 agriculture, 3 rangeland, 4 forest, 5 water, 6 barren; Inria 0 background,
1 building; URUR 0 others, 1 building, 2 farmland, 3 greenhouse, 4 woodland,
5 bareland, 6 water, 7 road. Ignore value is 255. Unknown DeepGlobe pixels
contribute to the confusion matrix, while class 0 is excluded from final mIoU.

For released DG/Inria ISDNet models, use the pinned upstream configuration
`configs/isdnet/isdnet_r18-d8_1224x1224_80k_DeepGlobe.py` or
`configs/isdnet/isdnet_r18-d8_2500x2500_40k_InriaAerial.py`. The public renderer
creates a JSON model configuration from that external checkout.
