import json
import os
from glob import glob
from uuid import uuid4

import fire
from helper import get_system_info


OUTPUT_DIR = "output"


def run(model_to_run, target_mpx: float = 0.5):
    """Run benchmark on test set.

    Args:
        model_to_run: paddle, deepsolo, etc.
        target_mpx: target megapixel value, 0.5 by default
    """
    print(f"model_to_run: {model_to_run}, target_mpx:{target_mpx}")

    text_dir = "imgs/test-set/text/resized/0.5"
    notext_dir = "imgs/test-set/notext/resized/0.5"
    if target_mpx == 0.33:
        # NOTE: Ensure that you run the following commands first to prepare the dataset
        # `python3 resize_images.py imgs/test-set/text 0.33`
        # `python3 resize_images.py imgs/test-set/notext 0.33`
        text_dir = "imgs/test-set/text/resized/0.33"
        notext_dir = "imgs/test-set/notext/resized/0.33"

    transaction_id = str(uuid4())
    target_dir = f"{OUTPUT_DIR}/{transaction_id}"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    # images with text
    text_file_list = glob(f"{text_dir}/*.jpg")
    if not text_file_list:
        print(f"no files found in {text_dir}")

    # images with NO-text
    notext_file_list = glob(f"{notext_dir}/*.jpg")
    if not notext_file_list:
        print(f"no files found in {notext_dir}")


    metrics = {}
    if model_to_run == "paddle":
        from ocr.paddle import Paddle

        paddle = Paddle(paddle_version="v1")
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v1"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v2")
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v2"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v3")
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v3"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v3-slim")
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v3-slim"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v3-ml-slim")
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v3-ml-slim"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v4")
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v4"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v2-mobile")
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v2-mobile"] = paddle_metrics
        del paddle


    elif model_to_run == "paddle_onnx":
        from ocr.paddle import Paddle

        paddle = Paddle(paddle_version="v1", use_onnx=True)
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v1-onnx"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v2", use_onnx=True)
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v2-onnx"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v3", use_onnx=True)
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v3-onnx"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v4", use_onnx=True)
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v4-onnx"] = paddle_metrics
        del paddle

        paddle = Paddle(paddle_version="v2-mobile", use_onnx=True)
        paddle_metrics = paddle.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["PaddleOCR-v2-mobile-onnx"] = paddle_metrics
        del paddle

    elif model_to_run == "paddle_det":
        from ocr.paddle_det import PaddleDet
        det_algorithm_ids = [
            "DB-r50",
            "DB-mobilenet",
            "DB++",
            "EAST-r50",
            "EAST-mobilenet",
            "FCE",
            "PSE-r50", # failed No such file or directory: 'ppocr/postprocess/pse_postprocess/pse'
            "PSE-mobilenet", # failed No such file or directory: 'ppocr/postprocess/pse_postprocess/pse'
            "SAST",
            "CT"
        ]
        for det_algorithm_id in det_algorithm_ids:
            paddle_det = PaddleDet(det_algorithm_id=det_algorithm_id)
            paddle_metrics = paddle_det.run_benchmark(
                text_file_list, notext_file_list, target_dir
            )
            metrics[f"PaddleOCR_det_{paddle_det.det_algorithm_id}"] = paddle_metrics
            del paddle_det

    elif model_to_run == "deepsolo":
        from ocr.deepsolo import DeepSolo

        deep_solo = DeepSolo()
        deepsolo_metrics = deep_solo.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["DeepSolo"] = deepsolo_metrics
        del deep_solo

    elif model_to_run == "east":
        from ocr.east import East

        east = East(target_mpx)
        east_metrics = east.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["east"] = east_metrics
        del east

    elif model_to_run == "opencv_db":
        from ocr.opencv_db import OpencvDB

        opencv_db = OpencvDB(model_id="DB_TD500_resnet18", target_mpx=target_mpx)
        opencv_db_metrics = opencv_db.run_benchmark(
            text_file_list, notext_file_list, target_dir
        )
        metrics["opencv_db"] = opencv_db_metrics
        del opencv_db

    elif model_to_run == "fast":
        from ocr.fast import Fast

        fast = Fast()
        fast_metrics = fast.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["fast"] = fast_metrics
        del fast

    elif model_to_run == "mmocr-DBNetR50":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("DBNetR50")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-DBNetR50"] = mmocr_metrics
        del mmocr

    elif model_to_run == "mmocr-DBNetR18":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("DBNetR18")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-DBNetR18"] = mmocr_metrics
        del mmocr

    elif model_to_run == "mmocr-DBNetpp":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("DBNetpp")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-DBNet++"] = mmocr_metrics
        del mmocr

    elif model_to_run == "mmocr-TextSnake":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("TextSnake")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-TextSnake"] = mmocr_metrics
        del mmocr
    
    elif model_to_run == "mmocr-PANet":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("PANet")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-PANet"] = mmocr_metrics
        del mmocr

    elif model_to_run == "mmocr-PSENet":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("PSENet")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-PSENet"] = mmocr_metrics
        del mmocr

    elif model_to_run == "mmocr-DRRG":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("DRRG")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-DRRG"] = mmocr_metrics
        del mmocr

    elif model_to_run == "mmocr-FCENet":
        from ocr.mmocr import MMOcr

        mmocr = MMOcr("FCENet")
        mmocr_metrics = mmocr.run_benchmark(text_file_list, notext_file_list, target_dir)
        metrics["mmocr-FCENet"] = mmocr_metrics
        del mmocr

    """
    # NOTE: Rest is useless, just keeping here as a reference.

    # NOTE: not better than PaddleOCR. RSS: 704876 kb.
    # TODO: PaddleOnnx does not draw bounding boxes yet.
    from ocr.paddle_onnx import PaddleOnnx
    paddle_onnx = PaddleOnnx()
    paddle_metrics = paddle_onnx.run_benchmark(text_file_list, notext_file_list, target_dir)
    metrics["PaddleOCRONNX"] = paddle_metrics
    del paddle_onnx

    # NOTE: super slow on CPU. 7s per image. RSS: 1417304 kb
    # TODO: does not draw bounding boxes yet.
    from ocr.easy_ocr import EasyOcr
    easy_ocr = EasyOcr()
    easyocr_metrics = easy_ocr.run_benchmark(text_file_list, notext_file_list, target_dir)
    metrics["EasyOCR"] = easyocr_metrics
    del easy_ocr

    # NOTE: uses craft, low-performant on CPU. 8s per image. RSS: 1140076 kb
    # TODO: does not draw bounding boxes yet.
    from ocr.keras_ocr import KerasOcr
    keras_ocr = KerasOcr()
    kerasocr_metrics = keras_ocr.run_benchmark(text_file_list, notext_file_list, target_dir)
    metrics["KerasOCR"] = kerasocr_metrics
    del keras_ocr
    """

    for key, val in metrics.items():
        print("{}\n{}\n\n".format(key, json.dumps(val, indent=2)))

    print(f"Output images can be found at {target_dir}")

    metrics["system_info"] = get_system_info()

    with open(f"{target_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(run)
