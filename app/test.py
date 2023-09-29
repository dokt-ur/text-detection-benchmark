import os
from glob import glob
from uuid import uuid4

from paddle_onnx import PaddleOnnx

paddle_onnx = PaddleOnnx()

OUTPUT_DIR = "output"

for img_dir in [
    "./imgs/test-set/text/resized/0.5",
    "./imgs/test-set/notext/resized/0.5",
    "./imgs/test-set/text/resized/0.33",
    "./imgs/test-set/notext/resized/0.33",
]:
    transaction_id = str(uuid4())
    target_dir = f"{OUTPUT_DIR}/{transaction_id}"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    text_file_list = glob(f"{img_dir}/*.jpg")
    paddle_onnx.run_benchmark(text_file_list, target_dir)

    print(f"Output images can be found at {target_dir} - Input: {img_dir}")
