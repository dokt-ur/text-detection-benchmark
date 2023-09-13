import os
import time

import cv2
import numpy as np
from helper import calculate_megapixels, resize_image_for_cvdnn
from imutils.object_detection import non_max_suppression
from ocr import Ocr
from tqdm import tqdm

TEST_IMG_PATH = "imgs/paddleocr-test-images/254.jpg"


class East(Ocr):
    def __init__(self, target_mpx):
        self.name = "east"
        self.test_image_path = TEST_IMG_PATH

        self.model_confidence = 0.5
        self.resize_image_width = 320  # should be multiple of 32
        self.resize_image_height = 320  # should be multiple of 32
        self.target_mpx = target_mpx

        self.model_file_path = "/root/github/text-detection-benchmark/benchmark/models/frozen_east_text_detection.pb"

        self.layer_names = [
            "feature_fusion/Conv_7/Sigmoid",  # output probabilities
            "feature_fusion/concat_3",  # can be used to derive the bounding box coordinates of text
        ]

        self.init_model()

    def init_model(
        self,
    ):
        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        self.model = cv2.dnn.readNet(self.model_file_path)

        # to eliminate cold start issue, run dummy inferences
        image = cv2.imread(self.test_image_path)
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (320, 320))

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.model.setInput(blob)
        (scores, geometry) = self.model.forward(self.layer_names)
        print(scores, geometry)

    def process_image(self, image: np.ndarray):
        (H, W) = image.shape[:2]
        print(H, W, H * W / 1e6)
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.model.setInput(blob)
        (scores, geometry) = self.model.forward(self.layer_names)

        (num_rows, num_cols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, num_rows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, num_cols):
                # if our score does not have sufficient probability, ignore it
                # if scoresData[x] < args["min_confidence"]:
                if scoresData[x] < 0.3:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # loop over the bounding boxes
        for startX, startY, endX, endY in boxes:
            # scale the bounding box coordinates based on the respective ratios
            # startX = int(startX * rW)
            # startY = int(startY * rH)
            # endX = int(endX * rW)
            # endY = int(endY * rH)

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

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
