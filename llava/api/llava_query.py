import json
import torch
import os
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from typing import List, Dict, Optional


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LLaVAQuery():
    def __init__(self, model_path: str, conv_mode: str = 'llava_llama_2', top_p: Optional[int] = None, temp: float = 0.2, num_beams: int = 1) -> None:
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)
        self.conv_mode = conv_mode
        self.top_p = top_p
        self.temp = temp
        self.num_beams = num_beams

    def query_one(self, image: Image.Image, system_msg: str, question: str, return_conv: bool = False) -> str:
        conversation = [{
            "from": "USER",
            "value": question
        }]
        return self.query_conv(image, system_msg, conversation, return_conv)
    
    def query_conv(self, image: Image.Image, system_msg: str, conversation: List[Dict[str, str]], return_conv: bool = False) -> str:
        """
        conversation: a list with json formatted dialogues.
                      roles should be specific to the conv mode chosen.
                      only first question should have image token.
                      last dialogue should always be human (here USER)
                      e.g.:
                      [{
                        "from": "USER"
                        "value": "<image> [[question 1]]"
                      },
                      {
                        "from": "ASSISTANT"
                        "value": "[[answer 1]]"
                      },
                      ...
                        "from": "USER"
                        "value": "<image> [[question N]]"
                      }
                      ]

        """
        conv = conv_templates[self.conv_mode].copy()
        conv.system = system_msg
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda()
        if self.model.config.mm_use_im_start_end:
            image_tk = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_tk = DEFAULT_IMAGE_TOKEN
        for dialogue in conversation:
            conv.append_message(dialogue['from'], dialogue['value'].replace('<image>', image_tk))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.temp,
                top_p=self.top_p,
                num_beams=self.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        answer = outputs.strip()
        if answer.endswith(stop_str):
            answer = answer[:-len(stop_str)]
        answer = answer.strip()
        if return_conv:
            ret_conv = conversation.copy()
            ret_conv.append({
                "from": conv.roles[1],
                "value": answer})
            return answer, ret_conv
        return answer
