# Evaluation

We provide some common pure text and multimodal benchmark evaluation codes for both Chinese and English. 
You can also simply modify the code and add more data evaluation benchmarks as needed.

## Data Preparation

We now support the use of the following benchmarks. 

- Pure Text
  
  - English
    - [MMLU](https://github.com/hendrycks/test)
    ```shell
    unzip mmlu.zip
    python evaluate/run.py --dataset_name mmlu --data_path ./evaluate/mmlu/mmlu/
    ```
    - [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh)
    ```shell
    unzip BBH.zip
    python evaluate/run.py --dataset_name bbh --data_path ./evaluate/bbh/BBH/
    ```
  - Chinese
    - [CMMLU](https://github.com/haonan-li/CMMLU)
    ```shell
    unzip cmmlu.zip
    python evaluate/run.py --dataset_name cmmlu --data_path ./evaluate/cmmlu/cmmlu/
    ```
    - [C-Eval](https://github.com/hkust-nlp/ceval/tree/main)
    ```shell
    unzip ceval.zip
    python evaluate/run.py --dataset_name ceval --data_path ./evaluate/ceval/ceval/formal_ceval/
    ```
    Your need to submit your c-eval results (i.e., result.json) to the online evaluation [website](https://cevalbenchmark.com/index.html).
  

- Multimodal

  - [SEED-Bench2](https://github.com/AILab-CVC/SEED-Bench/tree/main/SEED-Bench-2)
  
  You can download the dataset from [here](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2/tree/main).
  ```shell
    python evaluate/run.py --dataset_name seed_bench --data_path ./evaluate/seed_bench2/seed_bench2/
  ```
  - [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
  
  You can download the dataset from [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) after sending email to data owner.
  ```shell
    python evaluate/run.py --dataset_name mme --data_path ./evaluate/mme/mme/
  ```
  You can retrieve the final score from calucation.py in ./evaluate/mme/mme/eval_tool.
  - [MMVet](https://github.com/yuweihao/MM-Vet/tree/main)
  
  You can download the dataset from [here](https://github.com/yuweihao/MM-Vet?tab=readme-ov-file).
  ```shell
    python evaluate/run.py --dataset_name mme --data_path ./evaluate/mme/mme/
  ```
  Your need to submit your mm-vet results (i.e., result.json) to the online evaluation [website](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator).
  - MMB
  - MMMU
  - CMMMU