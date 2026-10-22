import os
import sys
from typing import Tuple, Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from .constants import T4_CONSTANTS, get_device
except ImportError:
    from constants import T4_CONSTANTS, get_device


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed_environment(
    rank: int,
    world_size: int,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    backend: str = "nccl",
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = T4_CONSTANTS.cudnn_benchmark
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    dist.init_process_group(
        backend=backend if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )


def init_distributed_mode(
    args,
    default_gpu_count: int = 2,
) -> Tuple[torch.device, int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device_type = "cuda" if torch.cuda.is_available() and not getattr(args, "force_cpu", False) else "cpu"

        if device_type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            torch.backends.cudnn.benchmark = T4_CONSTANTS.cudnn_benchmark
            try:
                dist.init_process_group(backend=T4_CONSTANTS.ddp_backend, device_id=device)
            except TypeError:
                dist.init_process_group(backend=T4_CONSTANTS.ddp_backend)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        is_master = (rank == 0)
        return device, rank, local_rank, world_size, is_master

    device = get_device()
    return device, 0, 0, 1, True


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    local_rank: int,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    if not is_dist_avail_and_initialized():
        return model.to(device)

    model = model.to(device)
    if device.type == "cuda":
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=False,
        )
    return DDP(model, find_unused_parameters=find_unused_parameters)


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def get_amp_components(device: torch.device, enabled: bool = True):
    use_cuda_amp = (device.type == "cuda" and enabled)

    class AutocastContextManager:
        def __enter__(self):
            if use_cuda_amp:
                self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                return self.ctx.__enter__()
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            if use_cuda_amp:
                return self.ctx.__exit__(exc_type, exc_val, exc_tb)
            return None

    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    return AutocastContextManager, scaler


def distributed_barrier():
    if is_dist_avail_and_initialized():
        if torch.cuda.is_available():
            try:
                dist.barrier(device_ids=[torch.cuda.current_device()])
            except TypeError:
                dist.barrier()
        else:
            dist.barrier()

