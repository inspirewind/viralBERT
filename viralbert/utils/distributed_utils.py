import os
import torch
import torch.distributed as dist

def setup_distributed():
    """初始化分布式训练环境"""
    if "LOCAL_RANK" not in os.environ:
        return -1

    local_rank = int(os.environ["LOCAL_RANK"])
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")
    
    return local_rank
