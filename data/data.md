# Data

To facilitate the application of this framework to training in different domains, we provide a script for adjusting the format of training data. 
Specifically, this framework can already support the training of [OCR](https://github.com/TencentARC/mllm-npu/tree/main/data/examples/ocr_example), [caption](https://github.com/TencentARC/mllm-npu/blob/main/data/examples/caption_example.tar), [interleaved image text](https://github.com/TencentARC/mllm-npu/blob/main/data/examples/interleaved_image_text_example.tar), and [pure text](https://github.com/TencentARC/mllm-npu/blob/main/data/examples/pure_text_example.jsonl). Among them, due to the large volume of caption and interleaved data, we chose [webdataset](https://github.com/webdataset/webdataset) to encapsulate it. 

We give a simple wds data construction method.

```shell
python ./data/process_wds.py
[2024-07-08 16:31:52.077279] start to write samples to shard ./tars/test-000001.tar
[2024-07-08 16:31:52.082296] complete to write samples to shard ./tars/test-000001.tar
[2024-07-08 16:31:52.072908] start to write samples to shard ./tars/test-000000.tar
[2024-07-08 16:31:52.088707] complete to write samples to shard ./tars/test-000000.tar
```

DataLoader example

```python
import webdataset as wds
from torch.utils.data import DataLoader

dataset = wds.WebDataset("./tars/test-{000000..000001}.tar")
dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

for ind, row in enumerate(dataloader):
    print(ind, row["jpg.txt"])
```

```shell
0 [b'text_1', b'text_2']
1 [b'text_4', b'text_3']
```