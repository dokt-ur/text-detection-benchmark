import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from mmcv import Config
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm import tqdm

FAST_BASE_DIR = "ext/FAST"
sys.path.append(FAST_BASE_DIR)
from helper import calculate_megapixels
from models import build_model
from models.utils import fuse_module, rep_model_convert
from ocr import Ocr


torch.set_num_threads(1)


warnings.filterwarnings("ignore")

TEST_IMG_PATH = (
    "imgs/test-set/text/test-10.webp"
)

MODELS_DIR = "models/FAST"


class Fast(Ocr):
    def __init__(
        self,
    ):
        self.name = "fast"
        self.use_gpu = False
        self.test_image_path = TEST_IMG_PATH

        self.min_score = None
        self.min_area = None
        self.batch_size = 1
        self.worker = 4
        self.use_ema = True

        self.config_file_path = f"{MODELS_DIR}/fast_tiny_tt_448_finetune_ic17mlt.py"
        self.model_weights_path = f"{MODELS_DIR}/fast_tiny_tt_448_finetune_ic17mlt.pth"

        self.init_model()

    def init_model(
        self,
    ):
        self.cfg = Config.fromfile(self.config_file_path)

        if self.min_score is not None:
            self.cfg.test_cfg.min_score = self.min_score
        if self.min_area is not None:
            self.cfg.test_cfg.min_area = self.min_area

        self.cfg.batch_size = self.batch_size
        self.cfg.use_cpu = not self.use_gpu

        self.cfg.test_cfg = dict(
            min_score=0.85,
            min_area=150,
            bbox_type="rect",
            result_path="outputs/submit_tt/",
        )

        # model
        self.model = build_model(self.cfg.model)

        if self.use_gpu:
            self.model = self.model.cuda()

        if os.path.isfile(self.model_weights_path):
            print(
                f"Loading model and optimizer from checkpoint '{self.model_weights_path}'"
            )
            sys.stdout.flush()
            checkpoint = torch.load(self.model_weights_path)

            if not self.use_ema:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["ema"]

            d = dict()
            for key, value in state_dict.items():
                tmp = key.replace("module.", "")
                d[tmp] = value
            self.model.load_state_dict(d)
        else:
            print(f"No checkpoint found at '{self.model_weights_path}'")
            raise

        self.model = rep_model_convert(self.model)

        # fuse conv and bn
        self.model = fuse_module(self.model)

        self.model.eval()

        # to eliminate cold start issue, run dummy inferences
        # use PIL, to be consistent with evaluation
        image = cv2.imread(self.test_image_path)
        file_name = self.test_image_path.split("/")[-1][:-4]

        self.process_image(image, file_name)

    @staticmethod
    def scale_aligned_short(img, short_size: int = 640):
        h, w = img.shape[0:2]
        scale = short_size * 1.0 / min(h, w)
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        return img

    @staticmethod
    def scale_aligned_long(img, long_size: int = 640):
        h, w = img.shape[0:2]
        scale = long_size * 1.0 / max(h, w)
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        return img

    def preprocess_image(self, img, file_name):
        img = img[:, :, [2, 1, 0]]
        # if img.shape[0] > img.shape[1]:
        #     img = cv2.transpose(img)
        #     img = cv2.flip(img, 0)
        img_meta = dict(org_img_size=np.array(img.shape[:2]))
        img = self.scale_aligned_short(
            img, short_size=448
        )  # min(img.shape[0], img.shape[1])
        # or
        # img = self.scale_aligned_long(img)
        img_meta.update(dict(img_size=np.array(img.shape[:2]), filename=file_name))

        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(img)

        data = dict(imgs=img, img_metas=img_meta)

        return data

    def process_image(self, image, file_name):
        data = self.preprocess_image(image, file_name)
        data["imgs"] = data["imgs"].unsqueeze(0).cpu()
        data.update(dict(cfg=self.cfg))

        bboxes = []
        scores = []
        with torch.no_grad():
            outputs = self.model(**data)
            # print(outputs)
            result = outputs["results"][0]
            bboxes = result["bboxes"]
            # scores = result["scores"]

        for bbox in bboxes:
            # print(bbox)
            tl = (bbox[2], bbox[3])
            br = (bbox[6], bbox[7])
            # cv2.polylines(output_image, [bbox], True, (255, 0, 0), 2)
            cv2.rectangle(image, tl, br, (255, 0, 0), 2)
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
            file_name = self.test_image_path.split("/")[-1][:-4]

            # process image
            start_time = time.time()
            bounded_image = self.process_image(image, file_name)
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
            file_name = self.test_image_path.split("/")[-1][:-4]

            # process image
            start_time = time.time()
            bounded_image = self.process_image(image, file_name)
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
    Fast()
