from mmtrain.config import ProjectConfig


def test_single_config_loads():
    config = ProjectConfig.from_yaml("configs/single.yaml")
    config.validate(world_size=1)
    assert config.model.total_sequence_length == 160


def test_tp_config_validates():
    config = ProjectConfig.from_yaml("configs/tp2.yaml")
    config.validate(world_size=2)

