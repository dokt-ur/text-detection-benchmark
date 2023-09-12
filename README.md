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
| EAST | 0.5 | "avg_per_image": 1989.99662399292<br>"avg_per_mpx": 4502.114396698061,<br>"avg_per_text_image": 1856.8594694137573,<br>"avg_per_notext_image": 2123.1337785720825,<br>"avg_per_text_mpx": 4635.6088868705665,<br>"avg_per_notext_mpx": 4368.619906525555,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 431532 | output/7be76222-4718-4a5c-a6e6-609f816b788c |
| EAST | 0.33 | "avg_per_image": 1337.0995879173279<br>"avg_per_mpx": 4428.828512567462,<br>"avg_per_text_image": 1299.4821310043335,<br>"avg_per_notext_image": 1374.7170448303223,<br>"avg_per_text_mpx": 4409.4488619947115,<br>"avg_per_notext_mpx": 4448.2081631402125,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 381884 | output/b2e8166c-a284-407e-bb10-b40d5839f01a |
| OpenCV-DB-TD500_resnet50 | 0.5 | "avg_per_image": 2970.6650614738464<br>"avg_per_mpx": 6744.270857472199,<br>"avg_per_text_image": 2717.5872087478638,<br>"avg_per_notext_image": 3223.742914199829,<br>"avg_per_text_mpx": 6854.008951073828,<br>"avg_per_notext_mpx": 6634.532763870571,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/cf47e43c-09ca-473a-b776-bd6ae66a5029 |
| OpenCV-DB-TD500_resnet50 | 0.33 | "avg_per_image": 2243.7480330467224<br>"avg_per_mpx": 7431.725060760914,<br>"avg_per_text_image": 2173.466420173645,<br>"avg_per_notext_image": 2314.0296459198,<br>"avg_per_text_mpx": 7376.830979164135,<br>"avg_per_notext_mpx": 7486.619142357694,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/965e6575-7514-4860-aae2-64733febb6f4 |
| OpenCV-DB-TD500_resnet18 | 0.5 | "avg_per_image": 1776.4896392822266<br>"avg_per_mpx": 4044.3756160032544,<br>"avg_per_text_image": 1663.5278463363647,<br>"avg_per_notext_image": 1889.4514322280884,<br>"avg_per_text_mpx": 4200.642521524172,<br>"avg_per_notext_mpx": 3888.1087104823364,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/9f133090-c1f5-47c1-a50e-013a9426669d |
| OpenCV-DB-TD500_resnet18 | 0.33 | "avg_per_image": 1363.336181640625<br>"avg_per_mpx": 4520.965167535967,<br>"avg_per_text_image": 1354.555869102478,<br>"avg_per_notext_image": 1372.116494178772,<br>"avg_per_text_mpx": 4598.402298058743,<br>"avg_per_notext_mpx": 4443.528037013193,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/f2a4b12e-d334-4282-bc20-620b0cff0024 |
| OpenCV-DB-IC15_resnet50 | 0.5 | "avg_per_image": 3120.8429098129272<br>"avg_per_mpx": 7064.494974426296,<br>"avg_per_text_image": 2851.475477218628,<br>"avg_per_notext_image": 3390.2103424072266,<br>"avg_per_text_mpx": 7151.247071040581,<br>"avg_per_notext_mpx": 6977.74287781201,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/897ac417-9df2-4a21-86be-9fff76b9ee7d |
| OpenCV-DB-IC15_resnet50 | 0.33 | "avg_per_image": 2262.6996517181396<br>"avg_per_mpx": 7511.839214257925,<br>"avg_per_text_image": 2279.9524307250977,<br>"avg_per_notext_image": 2245.4468727111816,<br>"avg_per_text_mpx": 7754.992912355277,<br>"avg_per_notext_mpx": 7268.685516160574,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/e80f7c42-7e1b-418c-84f9-97f65a35fc5b |
| OpenCV-DB-IC15_resnet18 | 0.5 | "avg_per_image": 1766.7537331581116<br>"avg_per_mpx": 4018.8696068749764,<br>"avg_per_text_image": 1644.4277048110962,<br>"avg_per_notext_image": 1889.079761505127,<br>"avg_per_text_mpx": 4151.7332120644805,<br>"avg_per_notext_mpx": 3886.006001685471,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/8e4a39c4-9136-4892-809a-07fce75c23cb |
| OpenCV-DB-IC15_resnet18 | 0.33 | "avg_per_image": 1338.9421820640564<br>"avg_per_mpx": 4446.29340852926,<br>"avg_per_text_image": 1325.110936164856,<br>"avg_per_notext_image": 1352.7734279632568,<br>"avg_per_text_mpx": 4514.680847264086,<br>"avg_per_notext_mpx": 4377.905969794435,<br>"total_text_images": 10,<br>"total_notext_images": 10 | N/A | output/31b64573-5362-4e72-99c2-245a66141042 |


System information:
```
"platform": "Linux"
"platform-release": "5.15.0-83-generic"
"architecture": "x86_64"
"processor": "x86_64"
"ram": "2 GB"
```