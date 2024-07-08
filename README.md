<!-- 
<center> 
MLLM-NPU 
</center> 
-->

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png">
</p>

#### <center>[[English]](./README.md) | [[中文]](./README_ZH.md)</center>

</br>

In recent years, the widespread use of NPUs has provided more training and usage resources for LLMs, especially MLLMs.
However, the current use of NPUs still has more or less adaptation issues.
Therefore, we provide a framework that can flexibly select different visual encoders, adapters, LLMs, and corresponding generation components to form MLLMs for training, inferring, and image generation.

For example, we give an implementation of a high-performance MLLM (i.e., SEED-X) using this framework. Of course, you can also choose different modules in this framework to build your own MLLM.

- [SEED-X](https://github.com/AILab-CVC/SEED-X/tree/main), a unified and versatile foundation model, which can serve as various multi-modal AI assistants **in the real world** after different instruction tuning, capable of responding to a variety of user needs through unifying **multi-granularity comprehension and generation**.

</br>

## 📢 News
**2024-07-08** 🔥 We release NPU-based multi-modal inference and pre-training code, and various ways to use SEED-X.

</br>

## 📋 TODOs
- [ ] Release more MLLMs on NPU.
- [ ] Multimodal benchmarks.

</br>

## 📃 Contents

- [Install](#🔨-install)
- [Demo](#💻-demo)
- [Model](#⚙️-Model)
- [Dataset](#🌐-dataset)
- [Train](#🏃-train)
- [Evaluation](#🌟-evaluation)

</br>

## 🔨 Install

- Dependencies & Environment
  - python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
  - [torch = 2.1.0+cpu](https://pytorch.org/) + [torch-npu = 2.1.0](https://pypi.org/project/torch-npu/2.1.0/)
  - ASCEND NPU (Recommend to use [910B]()) + [CANN](https://www.hiascend.com/en/software/cann)
    - CANN version
    
    </br>

    ```bash
    > cat /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info 
    package_name=Ascend-cann-toolkit
    version=8.0.T6
    innerversion=V100R001C17B214
    compatible_version=[V100R001C15,V100R001C18],[V100R001C30],[V100R001C13],[V100R003C11],[V100R001C29],[V100R001C10]
    arch=x86_64
    os=linux
    path=/usr/local/Ascend/ascend-toolkit/8.0.T6/x86_64-linux
    ```

- Installation
  - Clone the repo and install dependent packages

  </br>

  ```bash
  git clone -
  cd -
  pip install -r requirements.txt
  ```

</br>

## 💻 Demo

### Quick Start

To quickly try out this framework, you can execute the following script.

```bash
# For image comprehension
python ./demo/img2txt_inference.py

# For image generation
python ./demo/txt2img_generation.py
```

### Gradio Web UI

To launch a Gradio demo locally, please run the following commands one by one. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server ONCE.

1. Launch a contoller

    ```bash
    python ./seedx/serve/controller.py --host 0.0.0.0 --port 10000
    ```

2. Launch a model worker

    ```bash
    python ./seedx/serve/model_worker.py --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --config ./demo/worker_config.json
    ```

3. Launch a gradio web app

    ```bash
    python ./seedx/serve/gradio_app.py
    ```

4. You can also use this service through API, see [demo](./demo/demo.ipynb) for the format.

    ```json
    {
        "input_text": "put your input text here",
        "image": "put your input image (base64)",
        "image_gen": False or True
    }
    ```


</br>

## ⚙️ Model

We mainly adopt the `GeneraliazedMultimodalModels` in `[mllm.py](./mllm_npu/models/mllm.py)` as the overall architecture, which contains three basic modules:
- (1) a **language model**, e.g., LLaMA-2.
- (2) a **projector** to project image features into language embeddings.
- (3) a **vision encoder**, e.g., ViT.


The MLLM is built according to the model config with `hydra.utils.instantiate`, and you can find some samples in [models](./mllm_npu/configs/models).

The [SEED-X](https://github.com/AILab-CVC/SEED-X) models additionaly contains an **output projector** to obtain the image embeddings for generating images.


## 🌐 Dataset

You can prepare your own data to pre-train or fine-tune your model. Specifically, we provide four different tasks and corresponding formats (please refer to the [examples](./data/)). In order to use the data more efficiently, we use [webdataset](https://webdataset.github.io/webdataset/) to organize the data. Besides, please refer to [data.yaml](./seed_npu/configs/dataset/pretrain_data.yaml) for the index of the data. You can adjust the data sampling rate and other settings by setting it in this file.

</br>

## 🏃 Train

You need to specify the **model config** and **data config** in the training scripts, such as `[scripts/mllm_llama3_8b_siglip_vit_pretrain.sh](./scripts/mllm_llama3_8b_siglip_vit_pretrain.sh)`.

```bash
bash scripts/mllm_llama3_8b_siglip_vit_pretrain.sh
```

</br>

## 🌟 Evaluation
coming soon

</br>

## 💡 Citation

If you find the work helpful, please consider citing:

- SEED-X

    ```bash
    @article{ge2024seed,
        title={SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation},
        author={Ge, Yuying and Zhao, Sijie and Zhu, Jinguo and Ge, Yixiao and Yi, Kun and Song, Lin and Li, Chen and Ding, Xiaohan and Shan, Ying},
        journal={arXiv preprint arXiv:2404.14396},
        year={2024}
    }
    ```

</br>

## 🔎 License
This project is under the Apache-2.0 License. For models built with LLaMA or Qwen models, please also adhere to their licenses!
</br>

## 👍 Acknowledgement

This project is developed based on the source code of [SEED-X]().



