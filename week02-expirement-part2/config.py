from typing import Union
from pathlib import Path
from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # data
    dataset_path: Path
    dataset_output_path: Path
    training_dataset_path: Path


    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "config.yaml", *args, **kwargs):
        with open(filename, 'r') as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)