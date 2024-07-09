import io
import base64
import gradio as gr
import requests
import json

from PIL import Image

title = ("""
# MLLM-NPU (SEEDX)

[[Github]](https://github.com/TencentARC/mllm-npu/tree/main)
[[Paper]](https://arxiv.org/abs/2404.14396)

Demo of the MLLM-NPU. 

* SEED-X was trained with English-only data. It may process with other languages due to the inherent capabilities from LLaMA, but might not stable.
""")

css = """
img {
  font-family: 'Helvetica';
  font-weight: 300;
  line-height: 2;  
  text-align: center;

  width: auto;
  height: auto;
  display: block;
  position: relative;
}
img:before { 
  content: " ";
  display: block;
  position: absolute;
  top: -10px;
  left: 0;
  height: calc(100% + 10px);
  width: 100%;
  background-color: rgb(230, 230, 230);
  border: 2px dotted rgb(200, 200, 200);
  border-radius: 5px;
}
img:after { 
  content: " ";
  display: block;
  font-size: 16px;
  font-style: normal;
  font-family: FontAwesome;
  color: rgb(100, 100, 100);

  position: absolute;
  top: 5px;
  left: 0;
  width: 100%;
  text-align: center;
}
"""


def request_from_worker(image, text, force_img_gen, chat_history):
    if not force_img_gen:
        pload = {
            "input_text": text,
            "image": "",
            "image_gen": False
        }

        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        pload["image"] = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        image_str = f'<img src="data:image/png;base64,{pload["image"]}" alt="user upload image" />'

        chat_history.append((image_str + "\n" + text, None))
    else:
        pload = {
            "input_text": text,
            "image": "",
            "image_gen": True
        }

        chat_history.append((text, None))

    response = requests.post(
        "http://localhost:40000/worker_generate",
        headers={'User-Agent': 'Client'},
        json=pload,
        stream=False,
        timeout=1000
    )

    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            tmp = json.loads(chunk.decode())
            if not force_img_gen:
                chat_history.append((None, tmp["text"]))
            else:
                chat_history.append((None,
                                     f'<img src="data:image/png;base64,{tmp["image"]}" alt="user upload image" />\n{tmp["text"]}'))

    return None, chat_history


if __name__ == '__main__':
    # examples_mix = [
    #     ['https://github.com/AILab-CVC/SEED-X/blob/main/demos/bank.png?raw=true', 'Can I conntect with an advisor on Sunday?'],
    #     ['https://github.com/AILab-CVC/SEED-X/blob/main/demos/ground.png?raw=true',
    #      'Is there anything in the image that can protect me from catching the flu virus when I go out? Show me the location.'],
    #     ['https://github.com/AILab-CVC/SEED-X/blob/main/demos/arrow.jpg?raw=true', 'What is the object pointed by the red arrow?'],
    #     ['https://github.com/AILab-CVC/SEED-X/blob/main/demos/shanghai.png?raw=true', 'Where was this image taken? Explain your answer.'],
    #     ['https://github.com/AILab-CVC/SEED-X/blob/main/demos/GPT4.png?raw=true', 'How long does it take to make GPT-4 safer?'],
    #     ['https://github.com/AILab-CVC/SEED-X/blob/main/demos/twitter.png?raw=true',
    #      'Please provide a comprehensive description of this image.'],
    # ]
    # examples_text = [
    #     ['I want to build a two story cabin in the woods, with many commanding windows. Can you show me a picture?'],
    #     ['Use your imagination to design a concept image for Artificial General Intelligence (AGI). Show me an image.'],
    #     [
    #         'Can you design an illustration for “The Three-Body Problem” to depict a scene from the novel? Show me a picture.'],
    #     [
    #         'My four year old son loves toy trains. Can you design a fancy birthday cake for him? Please generate a picture.'],
    #     [
    #         'Generate an image of a portrait of young nordic girl, age 25, freckled skin, neck tatoo, blue eyes 35mm lens, photography, ultra details.'],
    #     ['Generate an impressionist painting of an astronaut in a jungle.']
    # ]

    with gr.Blocks(css=css) as demo:
        gr.Markdown(title)
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    image = gr.Image(type='pil', label='input_image')
                with gr.Row():
                    text = gr.Textbox(lines=12,
                                      label='input_text',
                                      elem_id='textbox',
                                      placeholder="Enter text and image, and press submit,", container=False)

                with gr.Row():
                    max_new_tokens = gr.Slider(minimum=64,
                                               maximum=1024,
                                               value=768,
                                               step=64,
                                               interactive=True,
                                               label="Max Output Tokens")

                with gr.Row():
                    force_img_gen = gr.Radio(choices=[True, False], value=False, label='Force Image Generation')

                with gr.Row():
                    # add_image_btn = gr.Button("Add Image")
                    # add_text_btn = gr.Button("Add Text")

                    submit_btn = gr.Button("Submit")

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_id='chatbot', label="MLLM-NPU", height=700)
                with gr.Row():
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        # with gr.Row():
        #     with gr.Column(scale=0.7):
        #         gr.Examples(examples=examples_mix, label='Input examples', inputs=[image, text], cache_examples=False)
        #     with gr.Column(scale=0.3):
        #         gr.Examples(examples=examples_text, label='Input examples', inputs=[text], cache_examples=False)

        # Register listeners
        # btn_list = [upvote_btn, downvote_btn, regenerate_btn, clear_btn]
        # upvote_btn.click(upvote_last_response, [dialog_state], [upvote_btn, downvote_btn])
        # downvote_btn.click(downvote_last_response, [dialog_state], [upvote_btn, downvote_btn])

        # regenerate_btn.click(regenerate, [dialog_state], [dialog_state, chatbot] + btn_list).then(
        #     http_bot, [dialog_state, input_state, max_new_tokens, max_turns, force_img_gen, force_bbox, force_polish],
        #     [dialog_state, input_state, chatbot] + btn_list)
        # add_image_btn.click(add_image, [dialog_state, input_state, image],
        #                     [dialog_state, input_state, image, chatbot])

        # add_text_btn.click(add_text, [dialog_state, input_state, text],
        #                    [dialog_state, input_state, text, chatbot] + btn_list)

        # submit_btn.click(
        #     add_image, [dialog_state, input_state, image], [dialog_state, input_state, image, chatbot] + btn_list)
        # .then(
        # add_text, [dialog_state, input_state, text],
        # [dialog_state, input_state, text, chatbot, upvote_btn, downvote_btn, regenerate_btn, clear_btn]).then(
        # http_bot,
        # [dialog_state, input_state, max_new_tokens, max_turns, force_img_gen, force_bbox, force_polish],
        # [dialog_state, input_state, chatbot] + btn_list)
        # clear_btn.click(clear_history, None, [dialog_state, input_state, chatbot] + btn_list)

        submit_btn.click(
            request_from_worker, [image, text, force_img_gen, chatbot], [image, chatbot]
        )

    demo.launch(server_name='0.0.0.0', share=False, server_port=12345)