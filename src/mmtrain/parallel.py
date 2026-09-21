from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

from .config import ProjectConfig


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def apply_tensor_parallel(model: nn.Module, config: ProjectConfig, device: torch.device) -> nn.Module:
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )
    except ImportError as exc:
        raise RuntimeError("当前 PyTorch 缺少 tensor.parallel；请安装 PyTorch 2.5+") from exc

    mesh = init_device_mesh(device.type, (config.parallel.tensor_parallel,), mesh_dim_names=("tp",))
    for layer in model.layers:
        parallelize_module(
            layer,
            mesh,
            {
                "attention.q_proj": ColwiseParallel(use_local_output=True),
                "attention.k_proj": ColwiseParallel(use_local_output=True),
                "attention.v_proj": ColwiseParallel(use_local_output=True),
                "attention.out_proj": RowwiseParallel(
                    input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True
                ),
                "feed_forward.up_proj": ColwiseParallel(use_local_output=True),
                "feed_forward.down_proj": RowwiseParallel(
                    input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=True
                ),
            },
        )
    return model


def apply_fsdp2(model: nn.Module, config: ProjectConfig, device: torch.device) -> nn.Module:
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError as exc:
        raise RuntimeError("当前 PyTorch 缺少 FSDP2；请安装 PyTorch 2.5+") from exc

    mesh = init_device_mesh(device.type, (config.parallel.data_parallel_shard,), mesh_dim_names=("fsdp",))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    return model


def maybe_wrap_model(model: nn.Module, config: ProjectConfig, device: torch.device) -> nn.Module:
    strategy = config.parallel.strategy
    if strategy == "tp":
        return apply_tensor_parallel(model, config, device)
    if strategy == "fsdp":
        return apply_fsdp2(model, config, device)
    if strategy == "ddp":
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    return model

