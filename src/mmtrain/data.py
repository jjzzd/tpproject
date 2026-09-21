from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DataConfig, ModelConfig
from .tokenizer import ByteTokenizer


def _conversation_to_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = " ".join(text_parts)
        lines.append(f"<{role}>\n{content}")
    return "\n".join(lines)


def normalize_record(record: dict[str, Any], text_field: str) -> tuple[str, list[str]]:
    if "messages" in record:
        text = _conversation_to_text(record["messages"])
    elif "prompt" in record and "response" in record:
        text = f"<user>\n{record['prompt']}\n<assistant>\n{record['response']}"
    else:
        text = str(record.get(text_field, ""))
    images = record.get("images", [])
    if "image" in record and not images:
        images = [record["image"]]
    if isinstance(images, str):
        images = [images]
    return text, [str(path) for path in images]


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    return (tensor - mean) / std


class JsonlMultimodalDataset(Dataset):
    def __init__(self, data: DataConfig, model: ModelConfig):
        self.model = model
        self.tokenizer = ByteTokenizer()
        self.image_root = Path(data.image_root)
        self.text_field = data.text_field
        path = Path(data.path)
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件: {path}")
        self.records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_number} 行格式错误") from exc
                text, images = normalize_record(record, self.text_field)
                if not text.strip():
                    raise ValueError(f"JSONL 第 {line_number} 行没有可训练文本")
                if len(images) > model.max_images:
                    images = images[: model.max_images]
                self.records.append({"text": text, "images": images})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        ids, text_mask = self.tokenizer.encode(record["text"], self.model.max_text_length)
        images = []
        for image_name in record["images"]:
            image_path = Path(image_name)
            if not image_path.is_absolute():
                image_path = self.image_root / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"找不到图片: {image_path}")
            images.append(_load_image(image_path, self.model.image_size))
        num_images = len(images)
        while len(images) < self.model.max_images:
            images.append(torch.zeros(3, self.model.image_size, self.model.image_size))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.bool),
            "pixel_values": torch.stack(images),
            "num_images": torch.tensor(num_images, dtype=torch.long),
        }


class SyntheticMultimodalDataset(Dataset):
    def __init__(self, data: DataConfig, model: ModelConfig, seed: int):
        self.samples = data.samples
        self.model = model
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        ids = torch.randint(
            4,
            self.model.vocab_size,
            (self.model.max_text_length,),
            generator=generator,
        )
        ids[0] = 1
        ids[-1] = 2
        images = torch.randn(
            self.model.max_images,
            3,
            self.model.image_size,
            self.model.image_size,
            generator=generator,
        )
        return {
            "input_ids": ids,
            "text_mask": torch.ones(self.model.max_text_length, dtype=torch.bool),
            "pixel_values": images,
            "num_images": torch.tensor(self.model.max_images, dtype=torch.long),
        }


def build_dataset(data: DataConfig, model: ModelConfig, seed: int) -> Dataset:
    if data.mode == "synthetic":
        return SyntheticMultimodalDataset(data, model, seed)
    if data.mode == "jsonl":
        return JsonlMultimodalDataset(data, model)
    raise ValueError(f"未知 data.mode: {data.mode}")

