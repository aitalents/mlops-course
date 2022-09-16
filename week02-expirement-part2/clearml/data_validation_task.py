from pathlib import Path
from clearml import Task, TaskTypes, Dataset
from config import AppConfig
from src.data_validation import main_actions


def main():
    task:Task = Task.init(project_name="itmo-mlops/experiment-part-2",
                     task_name="data validation", task_type=TaskTypes.data_processing)
    clearml_params = {
        "dataset_id":"154c3168ba74424bba3fd9a848a00594"
    }
    task.connect(clearml_params)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config: AppConfig = AppConfig.parse_raw()
    config.dataset_path = Path(dataset_path)
    main_actions(config=config)


if __name__ == "__main__":
    main()
