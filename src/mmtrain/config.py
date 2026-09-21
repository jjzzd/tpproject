from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 260
    num_layers: int = 4
    hidden_size: int = 256
    num_attention_heads: int = 8
    intermediate_size: int = 768
    max_text_length: int = 128
    max_images: int = 2
    image_size: int = 64
    patch_size: int = 16
    dropout: float = 0.0

    @property
    def patches_per_image(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def image_token_count(self) -> int:
        return self.max_images * self.patches_per_image

    @property
    def total_sequence_length(self) -> int:
        return self.image_token_count + self.max_text_length


@dataclass
class DataConfig:
    mode: str = "synthetic"
    path: str = "data/raw/train.jsonl"
    image_root: str = "data/raw"
    text_field: str = "text"
    samples: int = 1024
    num_workers: int = 0
    shuffle: bool = True


@dataclass
class ParallelConfig:
    strategy: str = "single"
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel_shard: int = 1
    pipeline_microbatches: int = 4


@dataclass
class OptimizerConfig:
    type: str = "fused_adamw"
    lr: float = 3.0e-4
    weight_decay: float = 0.1
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])


@dataclass
class TrainingConfig:
    output_dir: str = "runs/default"
    seed: int = 42
    steps: int = 20
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    dtype: str = "bfloat16"
    activation_checkpointing: bool = True
    grad_clip: float = 1.0
    log_interval: int = 1
    checkpoint_interval: int = 0
    compile: bool = False


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            parallel=ParallelConfig(**raw.get("parallel", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def validate(self, world_size: int) -> None:
        m, p, o = self.model, self.parallel, self.optimizer
        if m.image_size % m.patch_size:
            raise ValueError("model.image_size 必须能被 model.patch_size 整除")
        if m.hidden_size % m.num_attention_heads:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        if p.strategy == "single" and world_size != 1:
            raise ValueError("single 配置必须用一个进程启动")
        expected = {
            "tp": p.tensor_parallel,
            "pp": p.pipeline_parallel,
            "fsdp": p.data_parallel_shard,
            "ddp": world_size,
        }.get(p.strategy, 1)
        if p.strategy not in {"single", "ddp", "tp", "pp", "fsdp"}:
            raise ValueError(f"未知并行策略: {p.strategy}")
        if expected != world_size:
            raise ValueError(
                f"{p.strategy} 配置需要 {expected} 个进程，当前为 {world_size}"
            )
        if p.strategy == "tp":
            if m.hidden_size % p.tensor_parallel:
                raise ValueError("hidden_size 必须能被 TP 度整除")
            if m.num_attention_heads % p.tensor_parallel:
                raise ValueError("attention heads 必须能被 TP 度整除")
            if m.intermediate_size % p.tensor_parallel:
                raise ValueError("intermediate_size 必须能被 TP 度整除")
        if p.strategy == "pp":
            if p.pipeline_parallel != 2:
                raise ValueError("当前工程版 PP 仅实现两个 stage")
            if m.num_layers < 2:
                raise ValueError("PP2 至少需要两层 Transformer")
            if p.pipeline_microbatches < 2:
                raise ValueError("PP 应至少使用两个 microbatch")
        if o.type not in {"fused_adamw", "adamw_8bit"}:
            raise ValueError(f"未知优化器: {o.type}")

