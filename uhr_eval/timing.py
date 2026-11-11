"""Complete-path wall-clock timing; no disk reads within the timed region."""
import time

def configure():
    import cv2,torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True

def measure(calls,decoded_images,warmup=2,repeats=5):
    import torch
    names=list(calls)
    with torch.no_grad():
        for _ in range(warmup):
            for rgb in decoded_images:
                for name in names:
                    prediction=calls[name](rgb);torch.cuda.synchronize();del prediction
        for repeat in range(repeats):
            for index,rgb in enumerate(decoded_images):
                offset=(index+repeat)%len(names);order=names[offset:]+names[:offset]
                for position,name in enumerate(order):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter();prediction=calls[name](rgb);torch.cuda.synchronize()
                    ms=(time.perf_counter()-start)*1000
                    if prediction.device.type!='cuda' or prediction.shape!=rgb.shape[:2]:raise ValueError('Timing endpoint mismatch')
                    yield {'path':name,'image':index,'repeat':repeat,'position':position,'ms':ms,
                           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                    del prediction
