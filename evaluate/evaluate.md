# Evaluation

We provide some common pure text and multimodal benchmark evaluation codes for both Chinese and English. 
You can also simply modify the code and add more data evaluation benchmarks as needed.

## Data Preparation

We now support the use of the following benchmarks. 
You can place the corresponding dataset under [./eval_data/](./eval_data/)

- Pure Text

  - [MMLU](https://github.com/hendrycks/test)
  - [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh)
  - [CMMLU](https://github.com/haonan-li/CMMLU)
  - [C-Eval](https://github.com/hkust-nlp/ceval/tree/main)

    
- Multimodal

  - SEED-Bench
  - MME
  - MMVet
  - MMB
  - MMMU
  - CMMMU


## Run Evaluation

```shell
python evaluate/run.py -dataset_name bbh -data_path ./eval_data/BBH
```