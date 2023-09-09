import os
import sys
import time
import multiprocessing as mp
from tqdm import tqdm

sys.path.append("/root/github/text-detection-benchmark/benchmark/")
sys.path.append("/root/github/text-detection-benchmark/benchmark/ext/DeepSolo")

import cv2
import numpy as np
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model

from detectron2.config import get_cfg, config
from helper import calculate_megapixels
from ocr import Ocr

import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
# from adet.data.augmentation import Pad



TEST_IMG_PATH = "/root/github/text-detection-benchmark/benchmark/imgs/test-set/text/test-10.webp"
mp.set_start_method("spawn", force=True)


class ViTAEPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        # each size must be divided by 32 with no remainder for ViTAE
        # self.pad = Pad(divisible_size=32)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            # image = self.pad.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

    
class DeepSolo(Ocr):
    def __init__(self,):

        self.name = "deepsolo"
        self.use_gpu = False
        self.test_image_path = TEST_IMG_PATH
        self.conf_threshold = 0.5
        self.config_file = "/root/github/text-detection-benchmark/benchmark/ext/DeepSolo/configs/ViTAEv2_S/pretrain/150k_tt_mlt_13_15_textocr.yaml"
        self.model_weights_path =  "/root/github/text-detection-benchmark/benchmark/ext/DeepSolo/models/tt_vitaev2-s_finetune_synth-tt-mlt-13-15-textocr.pth"

        self.init_model()

    def init_model(
        self,
    ):
        cfg = get_cfg()

        # TODO: enable
        # cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.model_weights_path
        cfg.MODEL.DEVICE = "cuda" if self.use_gpu else "cpu"
        print(cfg.MODEL.DEVICE)
        
        # Set score_threshold for builtin models
        # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
        # cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
        # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.freeze()
        self.predictor = ViTAEPredictor(cfg)
        
        # to eliminate cold start issue, run dummy inferences

        # use PIL, to be consistent with evaluation
        image = read_image(self.test_image_path, format="BGR")
        predictions = self.predictor(image)
        
        start_time = time.time()
        predictions = self.predictor(image)
        print(
            "{}: detected {} instances in {:.2f}s".format(
                self.test_image_path, len(predictions["instances"]), time.time() - start_time
            )
        )
        print(predictions)


        out_filename = os.path.join("output_dir/", os.path.basename(self.test_image_path))
        # visualized_output.save(out_filename)

        
    def process_image(self, image: np.ndarray):
        predictions = self.predictor(image)
        print(predictions)
        #boxes = []
        #for row in predictions:
            # TODO: implement this
            #boxes.append([[int(row[0][0]), int(row[0][1])], [int(row[2][0]), int(row[2][1])]])
            # rect: [box[0][1]:box[1][1], box[0][0]:box[1][0]]
            #cv2.rectangle(image, (int(row[0][0]),int(row[0][1])), (int(row[2][0]),int(row[2][1])), (255, 0, 0), 2)
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


if __name__ == "__main__":
    DeepSolo()