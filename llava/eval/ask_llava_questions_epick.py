import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import tarfile
import io
from time import sleep

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

    episodes = json.load(open(args.val_file, 'r'))

    query_template = f"<image>/nDescribe this image using this list of potential objects in the list: <obj_list>. Not all these objects might appear in the image because some might be omitted. You are not allowed to define new objects. You should describe what are the status of these objects and how they are positioned with respect to each other. You should also answer what action is being performed by the human. "

    to_test = {
        "video_id": "P18_02",
        "tar_path": "/share/portal/ys749/EPIC-KITCHENS/P18/rgb_frames_val/P18_02.tar",
        "frames": [],
        "images_to_load": []
    } 
    for ep in episodes:
        if "P18_02" in ep["ep_id"]:
            to_test["frames"].append(ep)
            to_test["images_to_load"].append(ep["start_image"])

    out_filepath = os.path.expanduser(args.out_file)
    if os.path.exists(out_filepath) and not args.full_restart:
        # load it from last time
        with open(out_filepath, "r") as f:
            results_json = json.load(f)
    else:
        if os.path.exists(out_filepath):
            print(f"Warning: there's already a result json file at {out_filepath}")
            if input("Are you sure you want to start from scratch? (y/n)").strip() != "y":
                sys.exit()
        results_json = {
            "meta_data": {
                "model": model_name,
                "sys_msg": conv_templates[args.conv_mode].system,
                "query_template": query_template,
            },
            "P18_02": {}
        }

        # pre-save before starting
        with open(out_filepath, "w") as fout:
            fout.write(json.dumps(results_json))

    tar_file = to_test["tar_path"]

    with tarfile.open(tar_file) as tf:
        tarinfos = [tf.getmember(image_file) for image_file in to_test["images_to_load"]]
        images = [tf.extractfile(tarinfo) for tarinfo in tarinfos]
        images = [image.read() for image in images]
        images = [Image.open(io.BytesIO(image)).convert('RGB') for image in images]

    for i in tqdm(range(len(to_test["frames"]))):
        frame_data = to_test["frames"][i]
        
        if frame_data["id"] not in results_json:
            image_to_use = images[i]

            query = query_template.replace("<obj_list>", str(frame_data["objects"]))

            if model.config.mm_use_im_start_end:
                image_tk = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            else:
                image_tk = DEFAULT_IMAGE_TOKEN
            
            conv = conv_templates[args.conv_mode].copy()

            image_tensors = [image_processor.preprocess(image_to_use, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda()]

            conv.append_message(conv.roles[0], query)

            this_gen = conv.copy()
            this_gen.append_message(this_gen.roles[1], None)
            prompt = this_gen.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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
            print("==============")
            print(f"action: {frame_data['action']}")
            print(f"objects {frame_data['objects']}")
            # input("check answer")

            # so that we can save multiple json, one per video
            results_json["P18_02"][frame_data["id"]] = {
                                            "tar_path": frame_data["tar_path"],
                                            "image": frame_data["image"],
                                            "start_image": frame_data["start_image"],
                                            "end_image": frame_data["end_image"],
                                            "action": frame_data["action"],
                                            "objects": frame_data["objects"],
                                            "description": outputs}

            with open(out_filepath, "w") as fout:
                fout.write(json.dumps(results_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--val-file", type=str, default="epick-val.json")
    parser.add_argument("--out-file", type=str, default="out.json")
    parser.add_argument("--full_restart", action="store_true")
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
