import os
import time

import cv2
import numpy as np
from helper import calculate_megapixels, resize_image_for_cvdnn
from ocr import Ocr
from tqdm import tqdm


TEST_IMG_PATH = "imgs/paddleocr-test-images/254.jpg"
MODELS_DIR = "models/OPENCVDB/"


class OpencvDB(Ocr):
    def __init__(self, model_id, target_mpx):
        self.target_mpx = target_mpx
        self.name = "opencv_DB"
        self.test_image_path = TEST_IMG_PATH
        self.weights_file_path = (f"{MODELS_DIR}/{model_id}.onnx")

        self.init_model()

    def init_model(self, run_dummy_inference: bool = True):
        # load the text detection model to memory
        print("loading DB text detector...")
        self.model = cv2.dnn_TextDetectionModel_DB(self.weights_file_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # set post-processing params
        self.model.setBinaryThreshold(0.3)
        self.model.setPolygonThreshold(0.5)
        self.model.setMaxCandidates(200)
        self.model.setUnclipRatio(2.0)

        # set input shape and normalization params
        self.model.setInputScale(1.0 / 255.0)
        self.model.setInputMean((122.67891434, 116.66876762, 104.00698793))

        if run_dummy_inference:
            # to eliminate cold start issue, run dummy inferences
            image = cv2.imread(self.test_image_path)
            image = resize_image_for_cvdnn(image, target_megapixels=self.target_mpx)
            (height, width) = image.shape[:2]
            self.model.setInputSize(width, height)

            self.process_image(image)

    def process_image(self, image: np.ndarray):
        rotated_rectangles, _ = self.model.detectTextRectangles(image)

        # loop over the bounding boxes
        for rotated_rectangle in rotated_rectangles:
            # print(type(rotated_rectangle))
            # print(rotated_rectangle)
            points = cv2.boxPoints(rotated_rectangle)  # error in opencv 4.5.4
            # print("points:", points)
            cv2.drawContours(image, [np.int0(points)], 0, (255, 0, 0), 2)
            # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

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

            image = resize_image_for_cvdnn(image, target_megapixels=self.target_mpx)
            (height, width) = image.shape[:2]

            # TODO: model inference does not work well on dynamic input shapes
            self.init_model(run_dummy_inference=False)
            self.model.setInputSize(width, height)

            #dummy inference
            self.process_image(image)
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

            image = resize_image_for_cvdnn(image, target_megapixels=self.target_mpx)
            (height, width) = image.shape[:2]

            # TODO: model inference does not work well on dynamic input shapes
            self.init_model(run_dummy_inference=False)
            self.model.setInputSize(width, height)

            # dummy inference
            self.process_image(image)

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
