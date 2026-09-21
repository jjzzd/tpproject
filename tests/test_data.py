import torch

from mmtrain.config import DataConfig, ModelConfig
from mmtrain.data import SyntheticMultimodalDataset


def test_synthetic_shapes():
    model = ModelConfig(max_text_length=32, max_images=2, image_size=32, patch_size=16)
    dataset = SyntheticMultimodalDataset(DataConfig(samples=3), model, seed=7)
    sample = dataset[0]
    assert sample["input_ids"].shape == (32,)
    assert sample["pixel_values"].shape == (2, 3, 32, 32)
    assert sample["text_mask"].dtype == torch.bool

