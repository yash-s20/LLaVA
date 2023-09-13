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

# Configure YAML to dump pretty multiline strings (https://github.com/yaml/pyyaml/issues/240#issuecomment-1096224358)
import yaml

SETTINGS_YAML_PATH = os.path.join(os.path.dirname(__file__), "latest_config.yaml")

def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
    if data.count('\n') > 0:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def make_all_dir(prompt_name, outputs_main_folder):
    main_folder_path = os.path.join(outputs_main_folder, prompt_name)
    safe_mkdir(main_folder_path)

    return main_folder_path

def query_llava_about_one_video(ep_id, all_data_dict, overall_conv_template, 
                                model_name, tokenizer, model, image_processor, 
                                outputs_folder, args, settings_dict):
    # create result json or load from last time
    out_filepath = os.path.join(outputs_folder, f'{settings_dict["prompt_name"]}_{ep_id}_desc.json')

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
                "sys_msg": settings_dict["prompt"]["system"],
                "query_template": settings_dict["prompt"]["query_template"],
            },
            ep_id: {}
        }

        # pre-save before starting
        with open(out_filepath, "w") as fout:
            fout.write(json.dumps(results_json))

    # determine what to use
    tar_path = ""
    frame_list = []
    images_to_load = []

    # TODO: currently only support loading one type of image
    image_type_to_use = settings_dict["images_to_use"][0]

    for ep in all_data_dict:
        if ep_id in ep["ep_id"]:
            if tar_path == "":
                tar_path = ep["tar_path"]
            frame_list.append(ep)
            images_to_load.append(ep[image_type_to_use])

    # load images ahead of time
    with tarfile.open(tar_path) as tf:
        tarinfos = [tf.getmember(image_file) for image_file in images_to_load]
        images = [tf.extractfile(tarinfo) for tarinfo in tarinfos]
        images = [image.read() for image in images]
        images = [Image.open(io.BytesIO(image)).convert('RGB') for image in images]

    if model.config.mm_use_im_start_end:
        image_tk = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_tk = DEFAULT_IMAGE_TOKEN

    query_template = settings_dict["prompt"]["query_template"]

    # start querying LLaVA
    for i in tqdm(range(len(frame_list))):
        frame_data = frame_list[i]
        
        conv = overall_conv_template.copy()
        raw_outputs = []
        full_description = ""

        if frame_data["id"] not in results_json[ep_id]:
            image_to_use = images[i]

            image_tensors = [image_processor.preprocess(image_to_use, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().cuda()]

            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"action: {frame_data['action']}")

            for j in range(len(query_template)):
                question = query_template[j]
                question = question.replace("<obj_list>", str(frame_data["objects"]))

                # add the quesiton to the conversation
                conv.append_message(conv.roles[0], question)

                this_gen = conv.copy()
                this_gen.append_message(this_gen.roles[1], None)
                prompt = this_gen.get_prompt()

                # print(prompt)
                # input("check prompt")

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

                print("-----------")
                print(f"Q: {question}")
                print(f"A: {outputs}")
                
                # add the prediction to the conversation log
                conv.append_message(conv.roles[1], outputs)
                
                # save the predicted outputs
                raw_outputs.append({
                    "question": question,
                    "answer": outputs
                })

                full_description=f"{full_description}{outputs}\n"

            # so that we can save multiple json, one per video
            results_json[ep_id][frame_data["id"]] = {
                                            "tar_path": frame_data["tar_path"],
                                            "image": frame_data["image"],
                                            "start_image": frame_data["start_image"],
                                            "end_image": frame_data["end_image"],
                                            "action": frame_data["action"],
                                            "objects": frame_data["objects"],
                                            "description": full_description.strip(), 
                                            "raw_predictions": raw_outputs}

            with open(out_filepath, "w") as fout:
                fout.write(json.dumps(results_json))


def query_llava_about_videos(args, settings_dict):
    outputs_folder = make_all_dir(settings_dict["prompt_name"], settings_dict["overall_outputs_folder"])

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(settings_dict["model_path"])
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(settings_dict["data_path"], "r") as fin:
        all_data_dict = json.load(fin)

    overall_conv_template = conv_templates[settings_dict["conv_mode"]].copy()
    overall_conv_template.replace_sys_msg(settings_dict["prompt"]["system"].strip())

    for ep_id in settings_dict["generate_from_id"]:
        query_llava_about_one_video(ep_id, all_data_dict, overall_conv_template, 
                                    model_name, tokenizer, model, image_processor, 
                                    outputs_folder, args, settings_dict)

if __name__ == "__main__":
    # 1) Read prompt information from YAML
    print("Reading from YAML file...")
    with open(SETTINGS_YAML_PATH, "r") as f:
        settings_dict = yaml.safe_load(f)

    if len(settings_dict["images_to_use"]) != 1:
        print("Error: currently only support adding one image")
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--full_restart", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    query_llava_about_videos(args, settings_dict)
