# text-detection-benchmark

#### Setup Instructions:

##### Enable virtual environment

```
python3 -m venv paddle_env
source paddle_env/bin/activate
```

##### Installation

```
cd benchmark
./bin/run
```

##### Resize images
```
cd benchmark
python3 resize_images.py imgs/test-set/text
python3 resize_images.py imgs/test-set/notext
```


##### Run bBnchmarks
```
cd benchmark
./bin/run
```

### Report


```
/usr/bin/time -v ./bin/run
```


| Model   | Performance | Max RSS (kbytes) | Output |
| -------- | ----------- | -------- | -------- | 
| PaddleOCR-v1 | "avg_per_image": 589.5262241363525<br>"avg_per_mpx": 1282.5902412509986,<br>"avg_per_text_image": 538.0283832550049,<br>"avg_per_notext_image": 641.0240650177002,<br>"avg_per_text_mpx": 1280.70864116116,<br>"avg_per_notext_mpx": 1284.4718413408373,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 692928 | output/4186453c-66ad-4b26-a64d-a0e240de182b |
| PaddleOCR-v2 | "avg_per_image": 648.6886382102966<br>"avg_per_mpx": 1417.4868263326116,<br>"avg_per_text_image": 605.3152084350586,<br>"avg_per_notext_image": 692.0620679855347,<br>"avg_per_text_mpx": 1448.2444248054685,<br>"avg_per_notext_mpx": 1386.7292278597547,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 727036 | output/9c9d6a30-96a2-4db7-8046-34cdc8301edd |
| PaddleOCR-v3 | "avg_per_image": 635.345983505249<br>"avg_per_mpx": 1383.2047744478243,<br>"avg_per_text_image": 594.1625595092773,<br>"avg_per_notext_image": 676.5294075012207,<br>"avg_per_text_mpx": 1410.7932359431736,<br>"avg_per_notext_mpx": 1355.6163129524753,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 735600 | output/1661e8d7-0625-41c3-87fd-e40e42a5fed5 |
| PaddleOCR-v3-slim | "avg_per_image": 638.0285739898682<br>"avg_per_mpx": 1390.6428260016712,<br>"avg_per_text_image": 590.9909009933472,<br>"avg_per_notext_image": 685.0662469863892,<br>"avg_per_text_mpx": 1408.5551103798637,<br>"avg_per_notext_mpx": 1372.7305416234788,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 670864 | output/1c7bef84-971a-4a9a-8a8a-d6aaee770948 |
| PaddleOCR-v3-slim-multilang | "avg_per_image": 694.6452379226685<br>"avg_per_mpx": 1510.1878428821476,<br>"avg_per_text_image": 617.3972845077515,<br>"avg_per_notext_image": 771.8931913375854,<br>"avg_per_text_mpx": 1473.6538487799826,<br>"avg_per_notext_mpx": 1546.7218369843124,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 675500 | output/38109c1d-e943-4a97-9a88-a8eecc7543f6 |
| PaddleOCR-v4 | "avg_per_image": 602.7003645896912<br>"avg_per_mpx": 1313.7928362300122,<br>"avg_per_text_image": 555.051589012146,<br>"avg_per_notext_image": 650.3491401672363,<br>"avg_per_text_mpx": 1324.4128859704847,<br>"avg_per_notext_mpx": 1303.1727864895395,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 727020 | output/5f94b41b-f11c-4725-99cc-e6789340046e |


System information:
```
"platform": "Linux"
"platform-release": "5.15.0-83-generic"
"architecture": "x86_64"
"processor": "x86_64"
"ram": "2 GB"
```