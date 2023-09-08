import fire 

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR,draw_ocr

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, rec=False) # need to run only once to download and load model into memory

def run(img_path: str):
    
    result = ocr.ocr(img_path, rec=False, cls=False)

    # read image using opencv python
    image = cv2.imread(img_path)

    for row in result[0]:
        # conver float to integer
        row = [[int(r[0]), int(r[1])] for r in row]
        # Draw polyline on image
        cv2.polylines(image, [np.array(row)], True, (255, 0, 0), 1)

    cv2.imwrite("out.jpg", image)

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)


    # draw result
    #result = result[0]
    #image = Image.open(img_path).convert('RGB')
    #boxes = [line[0] for line in result]
    #txts = [line[1][0] for line in result]
    #scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    #im_show = Image.fromarray(im_show)
    #im_show.save('result.jpg')

if __name__ == "__main__":
    fire.Fire(run)
    