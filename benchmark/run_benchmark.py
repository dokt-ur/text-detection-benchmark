import os
import time
from glob import glob
from uuid import uuid4

import cv2
import fire
import numpy as np
from helper import generate_html_report
from paddleocr import PaddleOCR
from tqdm import tqdm

OUTPUT_DIR = "output"
OCR_LANG = "en"
USE_GPU = False
USE_ANGLE_CLASSIFIER = True
DEBUG_MODE = True

CUSTOM_MODEL_PATH = "models/"
TEST_IMG_PATH = "imgs/paddleocr-test-images/254.jpg"

# initialize paddle ocr only once to download and load model into memory
# ocr = PaddleOCR(use_angle_cls=USE_ANGLE_CLASSIFIER, lang=OCR_LANG, use_gpu=USE_GPU)

# ref : https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_ch/inference.md

ocr = PaddleOCR(
    det_model_dir=CUSTOM_MODEL_PATH,
    rec=False,
    use_angle_cls=USE_ANGLE_CLASSIFIER,
    use_gpu=USE_GPU,
    lang=OCR_LANG,
)

# to eliminate cold start issue, run dummy inferences
result = ocr.ocr(
    TEST_IMG_PATH,
    rec=False,
)
result = ocr.ocr(
    TEST_IMG_PATH,
    rec=False,
)


def run():
    """Run benchmark on test set."""

    METRIC_TP = 0
    METRIC_TN = 0
    METRIC_FP = 0
    METRIC_FN = 0
    elapsed_seconds = []

    text_dir = "imgs/test-set/text/resized"
    notext_dir = "imgs/test-set/notext/resized"

    # images with text
    text_file_list = glob(f"{text_dir}/*.jpg")

    if not text_file_list:
        print(f"no files found in {text_dir}")
        return

    transaction_id = str(uuid4())
    target_dir = f"{OUTPUT_DIR}/{transaction_id}/text"
    if not os.path.exists(target_dir):
        print(f"created emtpy dir: {target_dir}")
        os.makedirs(target_dir)

    for file_path in tqdm(text_file_list):
        # read image into memory
        image = cv2.imread(file_path)

        # process image
        start_time = time.time()
        result = ocr.ocr(image, rec=False, cls=False)
        elapsed_time = time.time() - start_time

        print(f"elapsed time: {elapsed_time}")
        elapsed_seconds.append(elapsed_time)

        # draw bounding boxes
        detected_text_regions = []
        for row in result:
            # print(row)
            # convert float to integer
            row = [[int(r[0]), int(r[1])] for r in row]
            # Draw polyline on image

            cv2.polylines(image, [np.array(row)], True, (255, 0, 0), 2)
            detected_text_regions.append(np.array(row))

        if DEBUG_MODE:
            file_id = os.path.split(file_path)[-1]
            output_file_path = f"{target_dir}/{file_id}_out.jpg"
            cv2.imwrite(output_file_path, image)

        if detected_text_regions:
            METRIC_TP += 1
        else:
            METRIC_FN += 1

    # images with NO-text
    notext_file_list = glob(f"{notext_dir}/*.jpg")

    if not notext_file_list:
        print(f"no files found in {notext_dir}")
        return

    transaction_id = str(uuid4())
    target_dir = f"{OUTPUT_DIR}/{transaction_id}/notext"
    if not os.path.exists(target_dir):
        print(f"created emtpy dir: {target_dir}")
        os.makedirs(target_dir)

    for file_path in tqdm(notext_file_list):
        # read image into memory
        image = cv2.imread(file_path)

        # process image
        start_time = time.time()
        result = ocr.ocr(image, rec=False, cls=False)
        elapsed_time = time.time() - start_time

        print(f"elapsed time: {elapsed_time}")
        elapsed_seconds.append(elapsed_time)

        # draw bounding boxes
        detected_text_regions = []
        for row in result:
            # print(row)
            # convert float to integer
            row = [[int(r[0]), int(r[1])] for r in row]

            # Draw polyline on image
            cv2.polylines(image, [np.array(row)], True, (255, 0, 0), 2)
            detected_text_regions.append(np.array(row))

        if DEBUG_MODE:
            file_id = os.path.split(file_path)[-1]
            output_file_path = f"{target_dir}/{file_id}_out.jpg"
            cv2.imwrite(output_file_path, image)

        if detected_text_regions:
            METRIC_FP += 1
        else:
            METRIC_TN += 1

    print(f"output images can be found at {target_dir}")

    total = sum(elapsed_seconds)
    average_ms = (total / len(elapsed_seconds)) * 1000
    print(f"avg (ms): {average_ms}")

    print(
        "METRICS: \n "
        + f"TEXT: \n TOTAL_IMAGE_COUNT: {len(text_file_list)}"
        + f"- text_detected:{METRIC_TP}, no_text_detected:{METRIC_FN} \n\n"
        + f"NO-TEXT: \n TOTAL_IMAGE_COUNT: {len(notext_file_list)}"
        + f"- text_detected:{METRIC_FP}, no_text_detected:{METRIC_TN} \n\n"
    )

    metrics = {
        "True positive": METRIC_TP,
        "False negative": METRIC_FN,
        "True negative": METRIC_TN,
        "False positive": METRIC_FP,
        "Total images with text": len(text_file_list),
        "Total images with no-text": len(notext_file_list),
        "average_ms": average_ms,
    }

    generate_html_report(transaction_id, OUTPUT_DIR, metrics)


if __name__ == "__main__":
    fire.Fire(run)
