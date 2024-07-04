import os
import sys
import logging
import logging.handlers
import torch
import requests
import deepspeed
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.deepspeed import is_deepspeed_zero3_enabled

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if handler is None:
        os.makedirs('.', exist_ok=True)
        filename = os.path.join('.', logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename,
                                                            when='D',
                                                            utc=True,
                                                            encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]
    }
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def reload_qwen_vit(model_path, output_path):
    torch.manual_seed(1234)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cpu", trust_remote_code=True).eval()

    visual_encoder = model.transformer.visual
    print(visual_encoder)

    torch.save(visual_encoder.state_dict(), output_path)


def remove_mismatched_weights(model, pretrained_state_dict):
    own_state = model.state_dict()
    mismatch_keys = []

    for name in list(pretrained_state_dict.keys()):
        if name not in own_state or own_state[
                name].shape != pretrained_state_dict[name].shape:
            mismatch_keys.append(name)
            pretrained_state_dict.pop(name)

    return pretrained_state_dict, mismatch_keys


def load_zero3_checkpoint(module: nn.Module,
                          state_dict,
                          prefix="",
                          error_msgs=[],
                          top=True):
    zero3_enabled = is_deepspeed_zero3_enabled()

    if not is_deepspeed_zero3_enabled():
        state_dict, mismatch_keys = remove_mismatched_weights(
            module, state_dict)
        info = module.load_state_dict(state_dict, strict=False)

        if len(mismatch_keys) > 0:
            print("shape mismatch keys: ", mismatch_keys)

        if len(info.missing_keys) > 0:
            missing_keys = [
                _ for _ in info.missing_keys if "llm.base_model" not in _
            ]
            if len(mismatch_keys) > 0:
                print("missing keys: ", info.missing_keys)

        if len(info.unexpected_keys) > 0:
            print("unexpected keys: ", info.unexpected_keys)

    else:
        local_metadata = {}
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)

        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            named_parameters = dict(
                module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [
                named_parameters[k] for k in state_dict.keys()
                if k in named_parameters
            ]
            params_name = [
                k for k in state_dict.keys() if k in named_parameters
            ]

            named_buffers = dict(
                module.named_buffers(prefix=prefix[:-1], recurse=False))
            buffers_to_gather = [
                named_buffers[k] for k in state_dict.keys()
                if k in named_buffers
            ]

            if len(params_to_gather) > 0 or len(buffers_to_gather) > 0:
                with deepspeed.zero.GatheredParameters(params_to_gather,
                                                       modifier_rank=0):
                    module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load_zero3_checkpoint(child,
                                      state_dict,
                                      prefix + name + ".",
                                      top=False)

        if top:
            if len(error_msgs) > 0:
                print('loading zero3 model weights meets error messages!')
                print(error_msgs)
            else:
                print('loading zero3 model weights success!')
