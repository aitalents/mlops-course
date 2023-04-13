import json
import numpy as np

import triton_python_backend_utils as pb_utils

from utils import non_max_suppression, scale_coords


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output_config = pb_utils.get_output_config_by_name(
            model_config, "output_post"
        )
        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        # model params
        self.scale_from = (640, 640)
        self.scale_to = (720, 1280)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = 0
        self.agnostic = True
        self.multi_label = False
        self.labels = ("human")
        self.max_det = 100

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            in_output = pb_utils.get_input_tensor_by_name(request, "output0").as_numpy()
            batch_size = in_output.shape[0]
            pred = non_max_suppression(in_output, self.conf_thres, self.iou_thres, self.classes, self.agnostic, max_det=self.max_det)
            # Process predictions
            # for i, det in enumerate(pred):  # per image
            #     if len(det):
            #         # Rescale boxes from img_size to im0 size
            #         det[:, :4] = scale_coords(self.scale_from, det[:, :4], self.scale_to).round()

            # print(f"Det: {det}")
            # out_shape = (batch_size, self.max_det, 5)
            # result_matrix = np.zeros(out_shape)
            # result_matrix.fill(-1)

            # for frame_idx, row in enumerate(pred):
            #     detection_idx = 0
            #     for box in row:
            #         if int(box[5]) != 0: # if labels[int(box[5])] != "person":
            #             continue

            #         result_matrix[frame_idx, detection_idx] = np.array([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])])
            #         detection_idx += 1

            # output = np.array(pred)
            output = np.array(pred)
            output = pb_utils.Tensor(
                "output_post", output.astype(self.output_dtype)
            )
            inference_response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(inference_response)

        return responses

