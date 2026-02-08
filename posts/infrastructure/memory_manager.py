import gc
import time
import torch


def cleanup_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(0.15)
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
    time.sleep(0.15)
    gc.collect()
