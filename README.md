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
| FAST-fast_tiny_tt_448_finetune_ic17mlt | 0.5 | "avg_per_image": 1649.969470500946<br>"avg_per_mpx": 3557.9852386399107,<br>"avg_per_text_image": 1469.2594289779663,<br>"avg_per_notext_image": 1830.6795120239258,<br>"avg_per_text_mpx": 3447.6984859130894,<br>"avg_per_notext_mpx": 3668.2719913667324,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 654336 | output/58f63d35-e8d8-4ec9-ad85-bf47a1e2c7ba |
| FAST-fast_tiny_tt_448_finetune_ic17mlt | 0.33 | "avg_per_image": 1057.3566913604736<br>"avg_per_mpx": 3301.990703331645,<br>"avg_per_text_image": 1091.094422340393,<br>"avg_per_notext_image": 1023.6189603805541,<br>"avg_per_text_mpx": 3493.921579275128,<br>"avg_per_notext_mpx": 3110.059827388162,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 876904 | output/0d9454c8-6f23-4190-9f99-92553582b2a3 |
| FAST-fast_tiny_tt_448_finetune_ic17mlt | fixed_short_size:448 | "avg_per_image": 890.9989714622498<br>"avg_per_mpx": 2012.1441881795877,<br>"avg_per_text_image": 892.8654193878174,<br>"avg_per_notext_image": 889.1325235366821,<br>"avg_per_text_mpx": 2242.6543831529025,<br>"avg_per_notext_mpx": 1781.6339932062722,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 867028 | output/7acb29c8-3db6-4ac6-bae8-cb9e11645e81 |
| MMOCR-DBNet-R50 | 0.5 | "avg_per_image": 12339.604592323303<br>"avg_per_mpx": 27295.570309540843,<br>"avg_per_text_image": 10266.860103607178,<br>"avg_per_notext_image": 14412.349081039429,<br>"avg_per_text_mpx": 25713.689047935855,<br>"avg_per_notext_mpx": 28877.451571145826,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1728936 | output/385e5be7-aed0-4f78-9a4f-359e54625643 |
| MMOCR-DBNet-R50 | 0.33 | "avg_per_image": 11775.07779598236<br>"avg_per_mpx": 36584.156148652866,<br>"avg_per_text_image": 10388.21907043457,<br>"avg_per_notext_image": 13161.936521530151,<br>"avg_per_text_mpx": 33179.211990572614,<br>"avg_per_notext_mpx": 39989.10030673312,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1743372 | output/21658fa1-ba9e-4168-aba1-a8392767e321 |
| MMOCR-DBNet-R18 | 0.5 | "avg_per_image": 2269.605028629303<br>"avg_per_mpx": 5063.868990297601,<br>"avg_per_text_image": 2065.7140731811523,<br>"avg_per_notext_image": 2473.4959840774536,<br>"avg_per_text_mpx": 5171.582007395035,<br>"avg_per_notext_mpx": 4956.155973200168,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1130028 | output/3c063bf4-49da-4605-8944-475c0c7b6028 |
| MMOCR-DBNet-R18 | 0.33 | "avg_per_image": 2203.944504261017<br>"avg_per_mpx": 6866.1470679610375,<br>"avg_per_text_image": 2082.623791694641,<br>"avg_per_notext_image": 2325.2652168273926,<br>"avg_per_text_mpx": 6667.528942841376,<br>"avg_per_notext_mpx": 7064.765193080701,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 856900 | output/3473fae2-ebd0-4cd4-b9dc-52af4c2032a1 |
| MMOCR-DBNet++ | 0.5 | "avg_per_image": 17882.421457767487<br>"avg_per_mpx": 39154.13013687761,<br>"avg_per_text_image": 13967.473149299622,<br>"avg_per_notext_image": 21797.36976623535,<br>"avg_per_text_mpx": 34632.66452851016,<br>"avg_per_notext_mpx": 43675.59574524505,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1740648 | output/9cb97396-712d-4fb4-a5a1-2579db9a771b |
| MMOCR-DBNet++ | 0.33 | "avg_per_image": 14545.601224899292<br>"avg_per_mpx": 45175.52733655203,<br>"avg_per_text_image": 12488.59658241272,<br>"avg_per_notext_image": 16602.605867385864,<br>"avg_per_text_mpx": 39907.33179089891,<br>"avg_per_notext_mpx": 50443.722882205155,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1734996 | output/d01d18b9-7e5c-48ea-9b33-96c7d1dfc878 |
| MMOCR-TextSnake | 0.5 | "avg_per_image": 8686.989796161652<br>"avg_per_mpx": 18981.324522789015,<br>"avg_per_text_image": 6532.720732688904,<br>"avg_per_notext_image": 10841.2588596344,<br>"avg_per_text_mpx": 16241.02805484744,<br>"avg_per_notext_mpx": 21721.620990730593,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1762396 | output/34fd90c7-50b5-498c-9252-8761a2668a30 |
| MMOCR-TextSnake | 0.33 | "avg_per_image": 8931.40082359314<br>"avg_per_mpx": 27670.316196898955,<br>"avg_per_text_image": 7215.968346595764,<br>"avg_per_notext_image": 10646.833300590515,<br>"avg_per_text_mpx": 22986.87912029674,<br>"avg_per_notext_mpx": 32353.753273501163,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1764728 | output/3f7d3ad8-107a-40c1-a371-4eda6aac3665 |
| MMOCR-PANet | 0.5 | "avg_per_image": 2875.3457069396973<br>"avg_per_mpx": 6427.218314716417,<br>"avg_per_text_image": 2721.468448638916,<br>"avg_per_notext_image": 3029.2229652404785,<br>"avg_per_text_mpx": 6784.677154121252,<br>"avg_per_notext_mpx": 6069.759475311583,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1483040 | output/4904ad78-60bf-4a92-84be-95f92ed3ded9 |
| MMOCR-PANet | 0.33 | "avg_per_image": 2905.3199887275696<br>"avg_per_mpx": 9036.925269436995,<br>"avg_per_text_image": 2798.879075050354,<br>"avg_per_notext_image": 3011.760902404785,<br>"avg_per_text_mpx": 8923.194385624456,<br>"avg_per_notext_mpx": 9150.656153249534,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1439892 | output/6acdd3d5-aa5b-4bc8-877f-78db5cf8b6ca |
| MMOCR-PSENet | 0.5 | "avg_per_image": 22147.805619239807<br>"avg_per_mpx": 51371.132029436216,<br>"avg_per_text_image": 23332.264947891235,<br>"avg_per_notext_image": 20963.34629058838,<br>"avg_per_text_mpx": 60735.4475209933,<br>"avg_per_notext_mpx": 42006.81653787914,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1759324 | output/c8b43b40-1a54-4d1c-a3bc-d448eb7f323c |
| MMOCR-PSENet | 0.33 | "avg_per_image": 26702.6474237442<br>"avg_per_mpx": 84517.9156802498,<br>"avg_per_text_image": 30040.35234451294,<br>"avg_per_notext_image": 23364.942502975464,<br>"avg_per_text_mpx": 98047.14025337299,<br>"avg_per_notext_mpx": 70988.69110712661,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1757936 | output/14751867-6235-4cdf-b210-d6d47cd99033 |
| MMOCR-DRRG | 0.5 | "avg_per_image": 4674.050319194794<br>"avg_per_mpx": 10606.791352334545,<br>"avg_per_text_image": 4837.121534347534,<br>"avg_per_notext_image": 4510.979104042053,<br>"avg_per_text_mpx": 12174.725674650048,<br>"avg_per_notext_mpx": 9038.857030019042,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1602336 | output/e76b408d-ef50-43de-8d32-ea3024bf39eb |
| MMOCR-DRRG | 0.33 | "avg_per_image": 4259.399425983429<br>"avg_per_mpx": 13311.866920033477,<br>"avg_per_text_image": 4469.461488723755,<br>"avg_per_notext_image": 4049.3373632431026,<br>"avg_per_text_mpx": 14321.273865184907,<br>"avg_per_notext_mpx": 12302.459974882051,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1588100 | output/278dd67e-dd54-458b-b2cc-a94351fca552 |
| MMOCR-FCENet | 0.5 | "avg_per_image": 6686.149978637695<br>"avg_per_mpx": 15097.849439652948,<br>"avg_per_text_image": 6445.410537719727,<br>"avg_per_notext_image": 6926.889419555664,<br>"avg_per_text_mpx": 16316.19546731014,<br>"avg_per_notext_mpx": 13879.503411995756,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1171344 | output/51a52859-9ce6-4863-b79e-244be8be9a21 |
| MMOCR-FCENet | 0.33 | "avg_per_image": 7263.067889213562<br>"avg_per_mpx": 22686.71404299577,<br>"avg_per_text_image": 7481.026911735535,<br>"avg_per_notext_image": 7045.108866691589,<br>"avg_per_text_mpx": 23969.436047702853,<br>"avg_per_notext_mpx": 21403.992038288678,<br>"total_text_images": 10,<br>"total_notext_images": 10 | 1251784 | output/115190db-9fef-437d-9ac9-542288795b5d |




System information:
```
"platform": "Linux"
"platform-release": "5.15.0-83-generic"
"architecture": "x86_64"
"processor": "x86_64"
"ram": "2 GB"
```