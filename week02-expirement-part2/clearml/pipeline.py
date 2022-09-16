from clearml import PipelineController
pipe = PipelineController(
  name="Training pipeline", project="itmo-mlops/experiment-part-2", version="0.0.1"
)
pipe.add_step(
    name='validation_data',
#    parents=['stage_data', ],
    base_task_project='itmo-mlops/experiment-part-2',
    base_task_name='data validation',
    parameter_override={
        'General/dataset_id': "154c3168ba74424bba3fd9a848a00594"},
)
pipe.add_step(
    name='training_step',
    parents=['preparation_data', ],
    base_task_project='itmo-mlops/experiment-part-2',
    base_task_name='training',
    parameter_override={
        'General/dataset_id': "${preparation_data.parameters.dataset_id}"},
)