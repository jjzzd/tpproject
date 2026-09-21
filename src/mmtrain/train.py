from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import save_checkpoint
from .config import ProjectConfig
from .data import build_dataset
from .model import MultimodalTransformer, PipelineStage0, PipelineStage1
from .parallel import distributed_context, maybe_wrap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal distributed trainer")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def slice_batch(batch: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in batch.items()}


def infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def build_optimizer(model: nn.Module, config: ProjectConfig) -> torch.optim.Optimizer:
    options = config.optimizer
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    kwargs = dict(
        lr=options.lr,
        weight_decay=options.weight_decay,
        betas=tuple(options.betas),
    )
    if options.type == "fused_adamw":
        return torch.optim.AdamW(parameters, fused=torch.cuda.is_available(), **kwargs)
    if options.type == "adamw_8bit":
        try:
            from torchao.optim import AdamW8bit
        except ImportError as exc:
            raise RuntimeError("adamw_8bit 需要安装可选依赖: pip install torchao") from exc
        return AdamW8bit(parameters, **kwargs)
    raise ValueError(options.type)


def model_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def log_metrics(output_dir: str, payload: dict, print_rank: bool) -> None:
    if not print_rank:
        return
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    path = Path(output_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train_regular(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterator[dict[str, torch.Tensor]],
    config: ProjectConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, config.training.steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        token_count = 0
        for _ in range(config.training.gradient_accumulation_steps):
            batch = move_batch(next(batches), device)
            loss = model(batch) / config.training.gradient_accumulation_steps
            loss.backward()
            loss_value += float(loss.detach())
            token_count += batch["input_ids"].numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        multiplier = world_size if config.parallel.strategy in {"ddp", "fsdp"} else 1
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
        )
        if world_size > 1:
            peaks = torch.tensor([peak_allocated, peak_reserved], device=device)
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            peak_allocated, peak_reserved = map(float, peaks)
        if step % config.training.log_interval == 0:
            log_metrics(
                config.training.output_dir,
                {
                    "step": step,
                    "loss": loss_value,
                    "seconds": elapsed,
                    "tokens_per_second": token_count * multiplier / elapsed,
                    "peak_allocated_gib_max_rank": peak_allocated,
                    "peak_reserved_gib_max_rank": peak_reserved,
                    "strategy": config.parallel.strategy,
                },
                rank == 0,
            )
        interval = config.training.checkpoint_interval
        if interval and step % interval == 0:
            save_checkpoint(
                config.training.output_dir,
                step,
                model,
                optimizer,
                config.parallel.strategy,
                rank,
                world_size,
            )


def train_pipeline(
    stage: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterator[dict[str, torch.Tensor]],
    config: ProjectConfig,
    device: torch.device,
    rank: int,
) -> None:
    stage.train()
    torch.cuda.reset_peak_memory_stats(device)
    micro = config.training.micro_batch_size
    count = config.parallel.pipeline_microbatches
    hidden_shape = (micro, config.model.total_sequence_length, config.model.hidden_size)
    dtype = model_dtype(config.training.dtype)

    for step in range(1, config.training.steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        full_batch = move_batch(next(batches), device)
        micro_batches = [slice_batch(full_batch, i * micro, (i + 1) * micro) for i in range(count)]
        loss_value = torch.zeros((), device=device, dtype=torch.float32)

        if rank == 0:
            outputs = []
            for batch in micro_batches:
                hidden = stage(batch)
                outputs.append(hidden)
                dist.send(hidden.detach().contiguous(), dst=1)
            for hidden in reversed(outputs):
                grad = torch.empty_like(hidden)
                dist.recv(grad, src=1)
                hidden.backward(grad)
        else:
            received = []
            losses = []
            for batch in micro_batches:
                hidden = torch.empty(hidden_shape, device=device, dtype=dtype)
                dist.recv(hidden, src=0)
                hidden.requires_grad_(True)
                loss = stage(hidden, batch) / count
                received.append(hidden)
                losses.append(loss)
                loss_value += loss.detach().float()
            for hidden, loss in reversed(list(zip(received, losses))):
                loss.backward()
                dist.send(hidden.grad.contiguous(), dst=0)

        torch.nn.utils.clip_grad_norm_(stage.parameters(), config.training.grad_clip)
        optimizer.step()
        dist.broadcast(loss_value, src=1)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_tensor = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device) / 1024**3,
                torch.cuda.max_memory_reserved(device) / 1024**3,
            ],
            device=device,
        )
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        if step % config.training.log_interval == 0:
            log_metrics(
                config.training.output_dir,
                {
                    "step": step,
                    "loss": float(loss_value),
                    "seconds": elapsed,
                    "tokens_per_second": micro * count * config.model.max_text_length / elapsed,
                    "peak_allocated_gib_max_rank": float(peak_tensor[0]),
                    "peak_reserved_gib_max_rank": float(peak_tensor[1]),
                    "strategy": "pp",
                    "schedule": "gpipe_fill_drain",
                },
                rank == 1,
            )
        interval = config.training.checkpoint_interval
        if interval and step % interval == 0:
            save_checkpoint(config.training.output_dir, step, stage, optimizer, "pp", rank, 2)


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    rank, local_rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError("多进程模式当前要求 NVIDIA CUDA GPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    config.validate(world_size)
    seed_everything(config.training.seed)

    dataset = build_dataset(config.data, config.model, config.training.seed)
    strategy = config.parallel.strategy
    sampler = None
    if strategy in {"ddp", "fsdp"}:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=config.data.shuffle)
    loader_batch = config.training.micro_batch_size
    if strategy == "pp":
        loader_batch *= config.parallel.pipeline_microbatches
    loader = DataLoader(
        dataset,
        batch_size=loader_batch,
        shuffle=config.data.shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    batches = infinite(loader)

    dtype = model_dtype(config.training.dtype) if device.type == "cuda" else torch.float32
    if strategy == "pp":
        stage: nn.Module = (
            PipelineStage0(config.model, config.training.activation_checkpointing)
            if rank == 0
            else PipelineStage1(config.model, config.training.activation_checkpointing)
        )
        stage = stage.to(device=device, dtype=dtype)
        optimizer = build_optimizer(stage, config)
        train_pipeline(stage, optimizer, batches, config, device, rank)
    else:
        model = MultimodalTransformer(config.model, config.training.activation_checkpointing)
        model = model.to(device=device, dtype=dtype)
        model = maybe_wrap_model(model, config, device)
        if config.training.compile:
            model = torch.compile(model)
        optimizer = build_optimizer(model, config)
        train_regular(model, optimizer, batches, config, device, rank, world_size)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
