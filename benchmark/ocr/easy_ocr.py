import os
import time

import cv2
import easyocr
import numpy as np
from helper import calculate_megapixels
from ocr import Ocr
from tqdm import tqdm

TEST_IMG_PATH = "imgs/paddleocr-test-images/254.jpg"

cv2.setNumThreads(1)


class EasyOcr(Ocr):
    def __init__(
        self,
    ):
        self.name = "easyOCR"
        self.lang = "en"
        self.use_gpu = False
        self.test_image_path = TEST_IMG_PATH

        self.init_model()

    def init_model(
        self,
    ):
        path = "models/EASYOCR/"

        self.model = easyocr.Reader(
            [self.lang],
            gpu=self.use_gpu,
            detect_network="craft",
            download_enabled=False,
            model_storage_directory=path,
        )

        # to eliminate cold start issue, run dummy inferences
        horizontal_list, free_list = self.model.detect(self.test_image_path)
        print(horizontal_list, free_list)

    def process_image(self, image: np.ndarray):
        horizontal_list, free_list = self.model.detect(image)
        # boxes = []
        # for row in result[0]:
        # boxes.append([[int(row[0][0]), int(row[0][1])], [int(row[2][0]), int(row[2][1])]])
        # rect: [box[0][1]:box[1][1], box[0][0]:box[1][0]]
        # cv2.rectangle(image, (int(row[0][0]),int(row[0][1])), (int(row[2][0]),int(row[2][1])), (255, 0, 0), 2)
        # row = [[int(r[0]), int(r[1])] for r in row]
        # cv2.polylines(image, [np.array(row)], True, (255, 0, 0), 2)
        return image

    def run_benchmark(self, text_file_list, notext_file_list, target_dir):
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
