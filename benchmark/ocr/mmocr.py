import os
import sys
import time
import warnings
from tqdm import tqdm

import cv2
import numpy as np
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox

from helper import calculate_megapixels
from ocr import Ocr


warnings.filterwarnings("ignore")

TEST_IMG_PATH = (
    "imgs/test-set/text/test-10.webp"
)

CONFIGS_BASE_DIR = "ext/mmocr/configs/"
MODEL_WEIGHTS_BASE_DIR = "models/MMOCR/"

# NOTE: models should be downloaded to base directory : https://mmocr.readthedocs.io/en/dev-1.x/modelzoo.html

class MMOcr(Ocr):
    def __init__(
        self,
        det_model_id: str = "DBNetR50"
    ):
        self.name = "mmocr"
        self.use_gpu = False
        self.device = "cuda" if self.use_gpu else "cpu"
        self.test_image_path = TEST_IMG_PATH

        self.det_model_id = None
        if det_model_id == "DBNetR50":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth"
        elif det_model_id == "DBNetR18":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth"
        elif det_model_id == "DBNetpp":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth"
        elif det_model_id == "TextSnake":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth"
        elif det_model_id == "PANet":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/panet/panet_resnet18_fpem-ffm_600e_ctw1500.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/panet_resnet18_fpem-ffm_600e_ctw1500_20220826_144818-980f32d0.pth"
        elif det_model_id == "PSENet":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/psenet/psenet_resnet50-oclip_fpnf_600e_ctw1500.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/psenet_resnet50-oclip_fpnf_600e_ctw1500_20221101_140406-d431710d.pth"
        elif det_model_id == "DRRG":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/drrg_resnet50_fpn-unet_1200e_ctw1500_20220827_105233-d5c702dd.pth"
        elif det_model_id == "FCENet":
            self.det_model_id = f"{CONFIGS_BASE_DIR}/textdet/fcenet/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py"
            self.det_model_weights_path = f"{MODEL_WEIGHTS_BASE_DIR}/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500_20220825_221510-4d705392.pth"
        else:
            raise ValueError("unsupported model!")
        
        self.min_confidence_score = 0.6

        self.predefined_conf = {
            "out_dir": f"results/{self.det_model_id}",
            "batch_size": 1,
            "show": False,
            "print_result": False,
            "save_pred": False,
            "save_vis": False,
        }

        self.init_model()

    def init_model(
        self,
    ):
        
        self.model = MMOCRInferencer(
            det=self.det_model_id,
            det_weights=self.det_model_weights_path,
            rec=None,
            rec_weights=None,
            kie=None,
            kie_weights=None,
            device=self.device
        )

        # to eliminate cold start issue
        image = cv2.imread(self.test_image_path)
        self.process_image(image)


    def process_image(self, image):

        conf = self.predefined_conf.copy()
        conf["inputs"] = image
        
        results = self.model(**conf)
        predictions = results["predictions"]
        for pred in predictions:
            det_polygons = pred["det_polygons"]
            det_scores = pred["det_scores"]
            det_bboxes = np.array([poly2bbox(poly) for poly in det_polygons])

            for score, bbox in zip(det_scores, det_bboxes):
                # print(score, bbox)
                if score > self.min_confidence_score:
                    [x1, y1, x2, y2] = bbox
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # cv2.polylines(output_image, [bbox], True, (255, 0, 0), 2)
        return image

    def run_benchmark(
        self,
        text_file_list,
        notext_file_list,
        target_dir,
    ):
        """Run paddle benchmark."""

        output_dir = f"{target_dir}/{self.name}/"
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


if __name__ == "__main__":
    MMOcr()
