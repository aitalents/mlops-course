from config import AppConfig
from clearml import Task, TaskTypes, Dataset
from pathlib import Path
from src.training import main_actions

def main():
    task:Task = Task.init(project_name="itmo-mlops/experiment-part-2",
                     task_name="training", task_type=TaskTypes.training)

    clearml_params = {
        "dataset_id":"977849d4a156461b97eafed3ae576670"
    }
    task.connect(clearml_params)
    dataset_path = Dataset.get(clearml_params["dataset_id"]).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.training_dataset_path = Path(dataset_path)
    main_actions(config=config)

if __name__ == "__main__":
    main()