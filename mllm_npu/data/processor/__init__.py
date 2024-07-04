import json
from .image_processing_clip import CLIPImageProcessor
from .image_processing_siglip import SiglipImageProcessor


def init_processor(processor_name, processor_json):
    processor_params = json.load(open(processor_json))
    if processor_name == "qwen_vit":
        processor = CLIPImageProcessor(**processor_params)
    elif processor_name == "siglip_vit":
        processor = SiglipImageProcessor(**processor_params)
    else:
        raise NotImplementedError()
    return processor
