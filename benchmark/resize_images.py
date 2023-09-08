import os
from glob import glob

import cv2
import fire
from helper import resize_image_to_target_megapixels
from tqdm import tqdm

OUTPUT_DIR = "resized"
TARGET_MEGAPIXELS = 0.5


def resize(file_dir: str):
    """Resize images in a given directory.

    Args:
        file_dir: absolute path. e.g. "/tmp/test_dir"
    """
    file_list = glob(f"{file_dir}/*.webp")

    if not file_list:
        print(f"no files found in {file_dir}")
        return

    target_dir = f"{file_dir}/{OUTPUT_DIR}"
    if not os.path.exists(target_dir):
        print(f"created emtpy dir: {target_dir}")
        os.makedirs(target_dir)

    for file_path in tqdm(file_list):
        file_id = os.path.split(file_path)[-1]
        output_file_path = f"{target_dir}/{file_id}_out.jpg"

        # read image using opencv python
        image = cv2.imread(file_path)

        print(f"file_path: {file_path}")

        # calculate megapixels
        output_image = resize_image_to_target_megapixels(image, TARGET_MEGAPIXELS)

        cv2.imwrite(output_file_path, output_image)


if __name__ == "__main__":
    fire.Fire(resize)
