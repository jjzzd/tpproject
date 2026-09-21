from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn


def save_checkpoint(
    output_dir: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    strategy: str,
    rank: int,
    world_size: int,
) -> None:
    root = Path(output_dir) / "checkpoints" / f"step-{step:08d}"
    if strategy in {"tp", "fsdp"}:
        import torch.distributed.checkpoint as dcp

        dcp.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint_id=str(root),
        )
        return
    if strategy == "pp":
        root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
            root / f"stage-{rank}.pt",
        )
        dist.barrier()
        return
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        target = model.module if hasattr(model, "module") else model
        torch.save(
            {"step": step, "model": target.state_dict(), "optimizer": optimizer.state_dict()},
            root / "trainer.pt",
        )
    if world_size > 1:
        dist.barrier()

