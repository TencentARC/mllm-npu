# Weights

You can organize the model weights in the following way, where most of the weights can be obtained in [weight](), and other components can also be obtained in the corresponding huggingface repo.

```shell
pretrained/
├── cvlm_llama2_tokenizer_100img_and_224loc_addpatch
│   ├── added_tokens.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.model
├── detokenizer
│   └── pytorch_model.bin
├── llama2
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model-00001-of-00006.bin
│   ├── pytorch_model-00002-of-00006.bin
│   ├── pytorch_model-00003-of-00006.bin
│   ├── pytorch_model-00004-of-00006.bin
│   ├── pytorch_model-00005-of-00006.bin
│   ├── pytorch_model-00006-of-00006.bin
│   └── pytorch_model.bin.index.json
├── pytorch_model.bin
├── stable-diffusion-xl-base-1.0
│   ├── 01.png
│   ├── comparison.png
│   ├── LICENSE.md
│   ├── model_index.json
│   ├── pipeline.png
│   ├── README.md
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── sd_xl_base_1.0_0.9vae.safetensors
│   ├── sd_xl_base_1.0.safetensors
│   ├── sd_xl_offset_example-lora_1.0.safetensors
│   ├── text_encoder
│   │   ├── config.json
│   │   ├── flax_model.msgpack
│   │   ├── model.fp16.safetensors
│   │   ├── model.onnx
│   │   ├── model.safetensors
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   ├── text_encoder_2
│   │   ├── config.json
│   │   ├── flax_model.msgpack
│   │   ├── model.fp16.safetensors
│   │   ├── model.onnx
│   │   ├── model.onnx_data
│   │   ├── model.safetensors
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── tokenizer_2
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── unet
│   │   ├── config.json
│   │   ├── diffusion_flax_model.msgpack
│   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   ├── diffusion_pytorch_model.safetensors
│   │   ├── model.onnx
│   │   ├── model.onnx_data
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   ├── vae
│   │   ├── config.json
│   │   ├── diffusion_flax_model.msgpack
│   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   └── diffusion_pytorch_model.safetensors
│   ├── vae_1_0
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.fp16.safetensors
│   │   └── diffusion_pytorch_model.safetensors
│   ├── vae_decoder
│   │   ├── config.json
│   │   ├── model.onnx
│   │   ├── openvino_model.bin
│   │   └── openvino_model.xml
│   └── vae_encoder
│       ├── config.json
│       ├── model.onnx
│       ├── openvino_model.bin
│       └── openvino_model.xml
```