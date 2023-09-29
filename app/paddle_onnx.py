import os
from copy import deepcopy

import cv2
import numpy as np
import onnxruntime as ort
from postprocess import build_post_process
from postprocess.utils import filter_tag_det_res, get_rotate_crop_image, sorted_boxes
from preprocess import create_operators, transform
from tqdm import tqdm

ONNX_MODELS_DIR = "models"


class PaddleOnnx(object):
    def __init__(
        self,
    ):
        self.name = "paddle_v4_onnx"
        self.model_file_path = f"{ONNX_MODELS_DIR}/en_PP-OCRv4_det.onnx"
        self.test_image_path = "./imgs/254.jpg"

        self.cpu_threads = 1  # TODO: implement
        self.total_process_num = 1  # TODO: implement

        self.init_model()

    def init_model(
        self,
    ):
        if not os.path.exists(self.model_file_path):
            raise Exception(f"model not found! {self.model_file_path}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.model = ort.InferenceSession(
            self.model_file_path, 
            providers=["CPUExecutionProvider"], 
            sess_options=sess_options
        )

        self.input_tensor = self.model.get_inputs()[0]
        self.output_tensors = None

        self.preprocess_op = create_operators()
        self.postprocess_op = build_post_process()

        # to eliminate cold start issue, run dummy inferences
        image = cv2.imread(self.test_image_path)

        self.process_image(image)

    def process_image(self, image: np.ndarray):
        ori_im = image.copy()

        img, shape_list = transform({"image": image}, self.preprocess_op)

        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)

        input_dict = {self.input_tensor.name: img}
        outputs = self.model.run(self.output_tensors, input_dict)

        preds = {"maps": outputs[0]}

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]

        dt_boxes = filter_tag_det_res(dt_boxes, ori_im.shape)

        sorted_dt_boxes = sorted_boxes(dt_boxes)
        # for bno in range(len(sorted_dt_boxes)):
        #    tmp_box = deepcopy(sorted_dt_boxes[bno])
        #    img_crop = get_rotate_crop_image(ori_im, tmp_box)

        # boxes = []
        for row in sorted_dt_boxes:
            # boxes.append([[int(row[0][0]), int(row[0][1])], [int(row[2][0]), int(row[2][1])]])
            # rect: [box[0][1]:box[1][1], box[0][0]:box[1][0]]
            # cv2.rectangle(image, (int(row[0][0]),int(row[0][1])), (int(row[2][0]),int(row[2][1])), (255, 0, 0), 2)
            row = [[int(r[0]), int(r[1])] for r in row]
            cv2.polylines(image, [np.array(row)], True, (255, 0, 0), 2)
        return image

    def run_benchmark(self, file_list, target_dir):
        """Run paddle benchmark."""

        output_dir = f"{target_dir}/{self.name}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_path in tqdm(file_list):
            # read image into memory
            image = cv2.imread(file_path)

            bounded_image = self.process_image(image)

            file_id = os.path.split(file_path)[-1]
            output_file_path = f"{output_dir}/{file_id}_out.jpg"
            cv2.imwrite(output_file_path, bounded_image)
