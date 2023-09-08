# text-detection-benchmark

#### setup instructions:

##### enable virtual environment

```
python3 -m venv paddle_env
source paddle_env/bin/activate
```

##### install requirements

```
cd benchmark
pip3 install -r requirements.txt
```

##### resize image
`python3 resize_images.py imgs/test-set/text`
`python3 resize_images.py imgs/test-set/notext`


##### run benchmark on images with text
`python3 test_paddle.py imgs/test-set/text/resized`


##### run benchmark on images with text
`python3 test_paddle.py imgs/test-set/notext/resized`



## [paddle quickstart](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/quickstart_en.md)
