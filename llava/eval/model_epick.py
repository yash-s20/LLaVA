import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import tarfile
import io

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    episodes = json.load(open(args.val_file, 'r'))
    out_file = os.path.expanduser(args.out_file)
    if os.path.dirname(out_file):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_file = open(args.out_file, 'w')
    results_json = []
    for episode in tqdm(episodes):
        out_list = []

        idx = episode["id"]
        tar_file = episode["tar_path"]
        # if "val" in args.val_file:
        #     tar_file = os.path.join(os.path.dirname(tar_file) + '_val', os.path.join(os.path.basename(tar_file)))
        queries = episode["conversations"]
        # print(idx, tar_file)
        # cur_prompt = qs
        if model.config.mm_use_im_start_end:
            image_tk = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_tk = DEFAULT_IMAGE_TOKEN
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        
        with tarfile.open(tar_file) as tf:
            image_files = [d['image'] for d in queries if "human" == d["from"]]
            tarinfos = [tf.getmember(image_file) for image_file in image_files]
            images = [tf.extractfile(tarinfo) for tarinfo in tarinfos]
            images = [image.read() for image in images]
            images = [Image.open(io.BytesIO(image)).convert('RGB') for image in images]
        conv = conv_templates[args.conv_mode].copy()
        # removing the header message because we are handling it as system message in conversation.py
        # conv.append_message(conv.roles[0], "We are a watching clips of a human washing dishes from an egocentric perspective. Provide what state was observed in the environment by the human and what action is being performed. Format as [state i]...\n[action i]...\n")
        # conv.append_message(conv.roles[1], "Sure! I'll be happy to help with that. Let's begin\n")
        image_tensors = [image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda() for image in images]
        for i, (dialogue, answer) in enumerate(zip([d for d in queries if "human"==d["from"]], [d for d in queries if "gpt"==d["from"]])):
            d = image_tk + dialogue["value"].replace("<image>", "")
            conv.append_message(conv.roles[0], d)
            this_gen = conv.copy()
            this_gen.append_message(this_gen.roles[1], None)
            prompt = this_gen.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            # print((input_ids == IMAGE_TOKEN_INDEX).sum()) # checking to make sure the images are right
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # print(prompt)
            # input("check prompt")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            print("----------------------prediction-----------------------")
            print(outputs)
            print("=============")
            print(answer["value"])

            # input("check answer")
            conv.append_message(conv.roles[1], outputs)
            out_list.append((outputs, answer["value"]))

        ans_id = shortuuid.uuid()
        
        # so that we can save multiple json, one per video
        results_json.append({"question_id": idx,
                                   "prompt": prompt,
                                   "predicted_vs_answer": out_list,
                                   "answer_id": ans_id,
                                   "model_id": model_name})

    out_file.write(json.dumps(results_json))
    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--val-file", type=str, default="epick-val.json")
    parser.add_argument("--out-file", type=str, default="out.json")
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
