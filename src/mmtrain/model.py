from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(value.dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape

        def shape(projected: torch.Tensor) -> torch.Tensor:
            local_heads = projected.size(-1) // self.head_dim
            return projected.view(batch, sequence, local_heads, self.head_dim).transpose(1, 2)

        query = shape(self.q_proj(hidden))
        key = shape(self.k_proj(hidden))
        value = shape(self.v_proj(hidden))
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(hidden)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
        return hidden * query_mask.unsqueeze(-1)


class MultimodalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_patch = nn.Conv2d(
            3,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.vision_norm = RMSNorm(config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.total_sequence_length, config.hidden_size)
        self.image_type = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.text_type = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch, image_count, channels, height, width = pixel_values.shape
        flat_images = pixel_values.reshape(batch * image_count, channels, height, width)
        patches = self.vision_patch(flat_images).flatten(2).transpose(1, 2)
        patches = patches.reshape(batch, -1, self.config.hidden_size)
        patches = self.vision_norm(patches) + self.image_type
        text = self.token_embedding(input_ids) + self.text_type
        hidden = torch.cat((patches, text), dim=1)
        positions = torch.arange(hidden.size(1), device=hidden.device)
        return hidden + self.position_embedding(positions)[None, :, :]


def build_masks(
    config: ModelConfig,
    text_mask: torch.Tensor,
    num_images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = text_mask.device
    batch = text_mask.size(0)
    image_slots = torch.arange(config.max_images, device=device)[None, :]
    valid_images = image_slots < num_images[:, None]
    image_mask = valid_images.repeat_interleave(config.patches_per_image, dim=1)
    query_mask = torch.cat((image_mask, text_mask), dim=1)
    sequence = query_mask.size(1)
    image_tokens = config.image_token_count
    query_index = torch.arange(sequence, device=device)[:, None]
    key_index = torch.arange(sequence, device=device)[None, :]
    image_query = query_index < image_tokens
    allowed_by_order = torch.where(image_query, key_index < image_tokens, key_index <= query_index)
    allowed = allowed_by_order[None, :, :] & query_mask[:, None, :]
    allowed = allowed & query_mask[:, :, None]
    return allowed[:, None, :, :], query_mask


def build_targets(config: ModelConfig, input_ids: torch.Tensor) -> torch.Tensor:
    prefix = torch.full(
        (input_ids.size(0), config.image_token_count),
        -100,
        dtype=torch.long,
        device=input_ids.device,
    )
    text_targets = input_ids.masked_fill(input_ids.eq(0), -100)
    return torch.cat((prefix, text_targets), dim=1)


class MultimodalTransformer(nn.Module):
    def __init__(self, config: ModelConfig, activation_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        self.embedding = MultimodalEmbedding(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.embedding(batch["pixel_values"], batch["input_ids"])
        attention_mask, query_mask = build_masks(
            self.config, batch["text_mask"], batch["num_images"]
        )
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                hidden = checkpoint(
                    layer,
                    hidden,
                    attention_mask,
                    query_mask,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, attention_mask, query_mask)
        logits = self.lm_head(self.norm(hidden))
        targets = build_targets(self.config, batch["input_ids"])
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=-100,
        )


class PipelineStage0(nn.Module):
    def __init__(self, config: ModelConfig, activation_checkpointing: bool):
        super().__init__()
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        split = math.ceil(config.num_layers / 2)
        self.embedding = MultimodalEmbedding(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(split)])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.embedding(batch["pixel_values"], batch["input_ids"])
        attention_mask, query_mask = build_masks(self.config, batch["text_mask"], batch["num_images"])
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, attention_mask, query_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, attention_mask, query_mask)
        return hidden


class PipelineStage1(nn.Module):
    def __init__(self, config: ModelConfig, activation_checkpointing: bool):
        super().__init__()
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        split = math.ceil(config.num_layers / 2)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers - split)]
        )
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        attention_mask, query_mask = build_masks(self.config, batch["text_mask"], batch["num_images"])
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, attention_mask, query_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, attention_mask, query_mask)
        logits = self.lm_head(self.norm(hidden))
        targets = build_targets(self.config, batch["input_ids"])
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=-100,
        )

