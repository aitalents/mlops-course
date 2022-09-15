from distutils.command.config import config
from statistics import mode
from clearml import Task, TaskTypes
from clearml import Dataset

import torch

from config import AppConfig
from train import main_actions as training_task
from train import validate_model


def main():
    task = Task.init(
        project_name="itmo-mlops",
        task_name="training-example",
        task_type=TaskTypes.data_processing
    )
    
    app_config = AppConfig.parse_raw()
    task.connect(app_config)
    
    clearml_config = {
        "dataset_id": app_config.dataset_id
    }
    
    dataset_path = Dataset.get(**clearml_config).get_local_copy()
    app_config.data_dir = dataset_path

    model = training_task(app_config)

    test_acc, test_loss = validate_model(model, app_config)

    torch.save(model.state_dict(), "models/model.pt")

    if test_acc >= app_config.val_acc_threshold and test_loss <= app_config.val_loss_threshold:
        task.upload_artifact("saved_model", "models/model.pt")


if __name__ == "__main__":
    main()
