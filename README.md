# text-detection-benchmark

#### Setup Instructions:

```
# NOTE: new virtual environments will be created

cd benchmark
./bin/install
```

##### Resize images before running the benchmark
```
cd benchmark
python3 resize_images.py imgs/test-set/text 0.33
python3 resize_images.py imgs/test-set/notext 0.33
```


##### Run benchmarks
```
cd benchmark
./bin/run
```

_If you receive the following error message:_
```
File "/root/github/text-detection-benchmark/envs/paddle/lib/python3.8/site-packages/paddle/fluid/core.py", line 269, in <module>
    from . import libpaddle
ImportError: libssl.so.1.1: cannot open shared object file: No such file or directory
```
```
# run these commands to fix SSL issues:
wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
```

<br/>
<hr/>
<br/>

### Report

| Model   | MPX | Performance | Max RSS (kbytes) | Output |
| -------- | ----- | ----------- | -------- | -------- | 
| PaddleOCR-v1 | 0.5 | "avg_per_image": 589.5262241363525<br>"avg_per_mpx": 1282.5902412509986,<br>"avg_per_text_image": 538.0283832550049,<br>"avg_per_notext_image": 641.0240650177002,<br>"avg_per_text_mpx": 1280.70864116116,<br>"avg_per_notext_mpx": 1284.4718413408373,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 692928 | output/4186453c-66ad-4b26-a64d-a0e240de182b |
| PaddleOCR-v1 | 0.33 | "avg_per_image": 422.01892137527466<br>"avg_per_mpx": 1315.9657813367428,<br>"avg_per_text_image": 413.8263940811157,<br>"avg_per_notext_image": 430.2114486694336,<br>"avg_per_text_mpx": 1324.885408547633,<br>"avg_per_notext_mpx": 1307.0461541258521,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 703080 | output/f74d6b9b-7910-4587-9d8b-d425be31c4c4 |
| PaddleOCR-v2 | 0.5 | "avg_per_image": 648.6886382102966<br>"avg_per_mpx": 1417.4868263326116,<br>"avg_per_text_image": 605.3152084350586,<br>"avg_per_notext_image": 692.0620679855347,<br>"avg_per_text_mpx": 1448.2444248054685,<br>"avg_per_notext_mpx": 1386.7292278597547,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 727036 | output/9c9d6a30-96a2-4db7-8046-34cdc8301edd |
| PaddleOCR-v2 | 0.33 | "avg_per_image": 459.4138503074646<br>"avg_per_mpx": 1431.7361249505846,<br>"avg_per_text_image": 450.1466751098633,<br>"avg_per_notext_image": 468.6810255050659,<br>"avg_per_text_mpx": 1439.5467884241696,<br>"avg_per_notext_mpx": 1423.9254614769998,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 730460 | output/7bee54f3-83a2-41f2-b625-5d11dc61a72d |
| PaddleOCR-v3 | 0.5 | "avg_per_image": 635.345983505249<br>"avg_per_mpx": 1383.2047744478243,<br>"avg_per_text_image": 594.1625595092773,<br>"avg_per_notext_image": 676.5294075012207,<br>"avg_per_text_mpx": 1410.7932359431736,<br>"avg_per_notext_mpx": 1355.6163129524753,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 735600 | output/1661e8d7-0625-41c3-87fd-e40e42a5fed5 |
| PaddleOCR-v3 | 0.33 | "avg_per_image": 462.34188079833984<br>"avg_per_mpx": 1441.4148733734528,<br>"avg_per_text_image": 456.45177364349365,<br>"avg_per_notext_image": 468.23198795318604,<br>"avg_per_text_mpx": 1460.2696648361948,<br>"avg_per_notext_mpx": 1422.560081910711,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 730424 | output/b07ba74d-997d-423a-ac1c-e4da6a9d384e |
| PaddleOCR-v3-slim | 0.5 | "avg_per_image": 638.0285739898682<br>"avg_per_mpx": 1390.6428260016712,<br>"avg_per_text_image": 590.9909009933472,<br>"avg_per_notext_image": 685.0662469863892,<br>"avg_per_text_mpx": 1408.5551103798637,<br>"avg_per_notext_mpx": 1372.7305416234788,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 670864 | output/1c7bef84-971a-4a9a-8a8a-d6aaee770948 |
| PaddleOCR-v3-slim | 0.33 | "avg_per_image": 547.1358299255371<br>"avg_per_mpx": 1705.4121365479589,<br>"avg_per_text_image": 525.7685899734497,<br>"avg_per_notext_image": 568.5030698776245,<br>"avg_per_text_mpx": 1683.6959020121326,<br>"avg_per_notext_mpx": 1727.1283710837845,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 675172 | output/f56c2e06-7737-4ec8-a6ff-317cf7e6b8f4 |
| PaddleOCR-v3-slim-multilang | 0.5 | "avg_per_image": 694.6452379226685<br>"avg_per_mpx": 1510.1878428821476,<br>"avg_per_text_image": 617.3972845077515,<br>"avg_per_notext_image": 771.8931913375854,<br>"avg_per_text_mpx": 1473.6538487799826,<br>"avg_per_notext_mpx": 1546.7218369843124,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 675500 | output/38109c1d-e943-4a97-9a88-a8eecc7543f6 |
| PaddleOCR-v3-slim-multilang | 0.33 | "avg_per_image": 491.35711193084717<br>"avg_per_mpx": 1531.4088914995339,<br>"avg_per_text_image": 485.5342388153076,<br>"avg_per_notext_image": 497.1799850463867,<br>"avg_per_text_mpx": 1552.330810555653,<br>"avg_per_notext_mpx": 1510.4869724434145,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 681732 | output/674515f0-3064-43ca-bc80-1a321de88fb5 |
| PaddleOCR-v4 | 0.5 | "avg_per_image": 602.7003645896912<br>"avg_per_mpx": 1313.7928362300122,<br>"avg_per_text_image": 555.051589012146,<br>"avg_per_notext_image": 650.3491401672363,<br>"avg_per_text_mpx": 1324.4128859704847,<br>"avg_per_notext_mpx": 1303.1727864895395,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 727020 | output/5f94b41b-f11c-4725-99cc-e6789340046e |
| PaddleOCR-v4 | 0.33 | "avg_per_image": 484.9559426307678<br>"avg_per_mpx": 1514.1375291429379,<br>"avg_per_text_image": 487.5907897949219,<br>"avg_per_notext_image": 482.32109546661377,<br>"avg_per_text_mpx": 1562.9058719256998,<br>"avg_per_notext_mpx": 1465.3691863601757,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 730204 | output/bea20211-c6cd-4e9f-b697-f650c4805a6a |
| EAST | 0.5 | "avg_per_image": 1929.5767784118652<br>"avg_per_mpx": 4003.488822925239,<br>"avg_per_text_image": 1971.1148023605347,<br>"avg_per_notext_image": 1888.0387544631958,<br>"avg_per_text_mpx": 4121.7865649008745,<br>"avg_per_notext_mpx": 3885.1910809496026,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 465036 | output/f95f1105-ebf1-4061-a929-01ded4f1e351 |
| EAST | 0.33 | "avg_per_image": 1342.130196094513<br>"avg_per_mpx": 4333.915858423202,<br>"avg_per_text_image": 1344.9748992919922,<br>"avg_per_notext_image": 1339.2854928970337,<br>"avg_per_text_mpx": 4334.916680345917,<br>"avg_per_notext_mpx": 4332.915036500489,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 393544 | output/9581f371-5e18-48cb-a6d0-7b869fea5776 |


System information:
```
"platform": "Linux"
"platform-release": "5.15.0-83-generic"
"architecture": "x86_64"
"processor": "x86_64"
"ram": "2 GB"
```