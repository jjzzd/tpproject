import torch

from mmtrain.config import DataConfig, ModelConfig
from mmtrain.data import SyntheticMultimodalDataset
from mmtrain.model import MultimodalTransformer


def test_forward_backward_cpu():
    config = ModelConfig(
        vocab_size=260,
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        max_text_length=16,
        max_images=1,
        image_size=32,
        patch_size=16,
    )
    sample = SyntheticMultimodalDataset(DataConfig(samples=1), config, seed=3)[0]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = MultimodalTransformer(config, activation_checkpointing=True)
    loss = model(batch)
    loss.backward()
    assert torch.isfinite(loss)

