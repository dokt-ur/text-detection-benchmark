import os
import time
from copy import deepcopy
from tqdm import tqdm

import cv2
import numpy as np
from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop

from helper import calculate_megapixels
from ocr import Ocr

# ref: https://github.com/PaddlePaddle/PaddleOCR/blob/0850586667308d38e113447e8d095e955092fe53/tools/infer/predict_det.py
TEST_IMG_PATH = "imgs/paddleocr-test-images/254.jpg"


class PaddleDet(Ocr):
    def __init__(self, det_algorithm_id: str = "DB-r50"):
        self.det_algorithm_id = det_algorithm_id

        self.paddle_version = "v3"
        self.name = "paddle"
        self.use_gpu = False
        self.lang = "en"
        self.custom_model_dir = "models/"
        self.test_image_path = TEST_IMG_PATH
        self.det_model_dir = "models/det/"
        self.det_box_type = "quad"
        self.max_batch_size = 30

        self.init_model()

    def init_model(
        self,
    ):
        self.ocr_version = None

        if self.paddle_version == "v3":
            self.ocr_version = "PP-OCRv3"
            if self.det_algorithm_id == "DB-r50":
                self.det_model_dir = "models/PADDLE/DB/det_r50_vd_db_v2_infer"
                self.det_algorithm = "DB"
            elif self.det_algorithm_id == "DB-mobilenet":
                self.det_model_dir = "models/PADDLE/DB/det_mv3_vd_db_v2_infer"
                self.det_algorithm = "DB"
            elif self.det_algorithm_id == "DB++":
                self.det_model_dir = "models/PADDLE/DBpp/det_r50_dbpp_td_tr_infer"
                self.det_algorithm = "DB++"
            elif self.det_algorithm_id == "EAST-r50":
                self.det_model_dir = "models/PADDLE/EAST/r50/det_r50_vd_east_v2.0_infer"
                self.det_algorithm = "DB++"
            elif self.det_algorithm_id == "EAST-mobilenet":
                self.det_model_dir = "models/PADDLE/EAST/mobilenet/det_mv3_east_v2.0_infer"
                self.det_algorithm = "EAST"
            elif self.det_algorithm_id == "FCE":
                self.det_model_dir = "models/PADDLE/FCE/det_r50_dcn_fce_ctw_v2.0_infer"
                self.det_algorithm = "FCE"
            elif self.det_algorithm_id == "PSE-r50":
                self.det_model_dir = "models/PADDLE/PSE/r50/det_r50_vd_pse_v2.0_infer"
                self.det_algorithm = "PSE"
            elif self.det_algorithm_id == "PSE-mobilenet":
                self.det_model_dir = "models/PADDLE/PSE/mobilenet/det_mv3_pse_v2.0_infer"
                self.det_algorithm = "PSE"
            elif self.det_algorithm_id == "SAST":
                self.det_model_dir = "models/PADDLE/SAST/det_r50_vd_sast_icdar15_v2.0_infer"
                self.det_algorithm = "SAST"
            elif self.det_algorithm_id == "CT":
                self.det_model_dir = "models/PADDLE/CT/det_r18_ct_infer"
                self.det_algorithm = "CT"

        class CustomConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        cfg = CustomConfig(
            det_algorithm=self.det_algorithm,
            det_model_dir=self.det_model_dir,
            use_gpu=self.use_gpu,
            use_npu=False,
            use_xpu=False,
            enable_mkldnn=False,
            det_box_type=self.det_box_type,
            lang=self.lang,
            use_onnx=False,
            det_limit_side_len = 960,
            det_limit_type = "max",
            benchmark=False,
            # DB
            det_db_thresh = 0.3,
            det_db_box_thresh =0.6,
            det_db_unclip_ratio=1.5,
            max_batch_size=10,
            use_dilation=False,
            det_db_score_mode="fast",
            # EAST
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            # SAST
            det_sast_score_thresh=0.5,
            det_sast_nms_thresh=0.2,
            # PSE
            det_pse_thresh=0,
            det_pse_box_thresh=0.85,
            det_pse_min_area=16,
            det_pse_scale=1,
            # FCE
            scales=[8, 16, 32],
            alpha=1.0,
            beta=1.0,
            fourier_degree=5
        )

        self.model = TextDetector(cfg)

        # to eliminate cold start issue, run dummy inferences
        image = cv2.imread(self.test_image_path)
        self.process_image(image)


    def process_image(self, image: np.ndarray):
        dt_boxes, _ = self.model(image)
        dt_boxes = self.sorted_boxes(dt_boxes)
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(image, [box], True, (255, 0, 0), 2)

        return image

    def run_benchmark(self, text_file_list, notext_file_list, target_dir):
        """Run paddle benchmark."""

        output_dir = f"{target_dir}/{self.name}_{self.paddle_version}_{self.det_algorithm_id}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        elapsed_seconds_text = []
        elapsed_seconds_notext = []
        elapsed_seconds_per_mpx_text = []
        elapsed_seconds_per_mpx_notext = []

        for file_path in tqdm(text_file_list):
            # read image into memory
            image = cv2.imread(file_path)

            # process image
            start_time = time.time()
            bounded_image = self.process_image(image)
            elapsed_seconds = time.time() - start_time

            elapsed_seconds_text.append(elapsed_seconds)
            elapsed_seconds_per_mpx_text.append(
                elapsed_seconds / calculate_megapixels(image)
            )

            file_id = os.path.split(file_path)[-1]
            output_file_path = f"{output_dir}/text_{file_id}_bounded.jpg"
            cv2.imwrite(output_file_path, bounded_image)

        for file_path in tqdm(notext_file_list):
            # read image into memory
            image = cv2.imread(file_path)

            # process image
            start_time = time.time()
            bounded_image = self.process_image(image)
            elapsed_seconds = time.time() - start_time

            elapsed_seconds_notext.append(elapsed_seconds)
            elapsed_seconds_per_mpx_notext.append(
                elapsed_seconds / calculate_megapixels(image)
            )

            file_id = os.path.split(file_path)[-1]
            output_file_path = f"{output_dir}/notext_{file_id}_bounded.jpg"
            cv2.imwrite(output_file_path, bounded_image)

        total_text_images = len(elapsed_seconds_text)
        total_notext_images = len(elapsed_seconds_notext)

        avg_per_image = (
            (sum(elapsed_seconds_text) + sum(elapsed_seconds_notext))
            / (total_text_images + total_notext_images)
        ) * 1000
        avg_per_text_image = (sum(elapsed_seconds_text) / total_text_images) * 1000
        avg_per_notext_image = (
            sum(elapsed_seconds_notext) / total_notext_images
        ) * 1000

        avg_per_mpx = (
            (sum(elapsed_seconds_per_mpx_text) + sum(elapsed_seconds_per_mpx_notext))
            / (total_text_images + total_notext_images)
        ) * 1000
        avg_per_text_mpx = (
            sum(elapsed_seconds_per_mpx_text) / total_text_images
        ) * 1000
        avg_per_notext_mpx = (
            sum(elapsed_seconds_per_mpx_notext) / total_notext_images
        ) * 1000

        metrics = {
            "avg_per_image": avg_per_image,
            "avg_per_mpx": avg_per_mpx,
            "avg_per_text_image": avg_per_text_image,
            "avg_per_notext_image": avg_per_notext_image,
            "avg_per_text_mpx": avg_per_text_mpx,
            "avg_per_notext_mpx": avg_per_notext_mpx,
            "total_text_images": total_text_images,
            "total_notext_images": total_notext_images,
        }

        return metrics

    @staticmethod
    def sorted_boxes(dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes