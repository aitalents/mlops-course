from typing import Union
from pathlib import Path
from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # data
    data_dir: str
    dataset_id: str
    # model training
    lr: float
    momentum: float
    num_epochs: int
    # validation
    val_acc_threshold: float
    val_loss_threshold: float

    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "config.yaml", *args, **kwargs):
        with open(filename, 'r') as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)