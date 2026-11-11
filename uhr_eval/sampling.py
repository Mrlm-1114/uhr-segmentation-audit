"""Deterministic shuffled epochs and per-occurrence augmentation for resuming."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def initialize_worker(worker_id):
    import cv2
    cv2.setNumThreads(0)


class SeededDataset(Dataset):
    def __init__(self,dataset,seed):
        self.dataset,self.seed=dataset,seed
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,item):
        index,occurrence=item
        seed=(self.seed+int(occurrence)*1000003)%(2**32)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        return self.dataset[index]


class TrainingBatches(Sampler):
    """Same shuffled/drop-last batches whether started from zero or a checkpoint."""
    def __init__(self,n,batch,iterations,start,seed):
        if n<batch or not 0<=start<=iterations:raise ValueError("Invalid batch schedule")
        self.n,self.batch,self.iterations,self.start,self.seed=n,batch,iterations,start,seed
    def __len__(self):
        return self.iterations-self.start
    def __iter__(self):
        steps=self.n//self.batch
        last_epoch=-1
        for iteration in range(self.start,self.iterations):
            epoch,offset=divmod(iteration,steps)
            if epoch!=last_epoch:
                order=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()
                last_epoch=epoch
            yield [(order[offset*self.batch+j],iteration*self.batch+j) for j in range(self.batch)]
