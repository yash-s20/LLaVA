# Installation
1. Go through [G2 Guide](https://docs.google.com/document/d/1t9RYGiZISpdbxU3UTOB4msIrUBUabL9KsNZEe8lY94U/edit?usp=sharing)
2. Cd into your folder in `/share/portal`, clone this LLaVA repo
3. Check out to the correct branch (see slack for details)
4. Follow [installation guide in this README](https://docs.google.com/document/d/1t9RYGiZISpdbxU3UTOB4msIrUBUabL9KsNZEe8lY94U/edit?usp=sharing) to install the `llava` conda environment
5. One-time set up
    - create a folder called `outputs` in the main directory of the LLaVA repo
    - in the `llava/eval/latest_config.yaml`, modify
        - the data path to be where you store the raw data: `epic-k_val_obj-list_data.json`. (Most likely that you just need to change hw575 to your netid)
        - the folder path to be where the output would be stored. (Most likely that you just need to change hw575 to your netid)


# Quick Start

## Prompt iteration/improvement
**Don't edit the prompt in the yaml file!** 

- Make a copy of the prompt in [the google drive](https://drive.google.com/drive/folders/1JxexTThMOMd-u6GIjwZ70DScV13_kebd?usp=sharing)
- Edit the prompt there (use Grammarly to check your grammar XD)
- Once you are happy with the prompt, paste it in `latest_settings.yml`

## Overall workflow
- It's helpful to first set `generate_all` to False, and generate results using one video (e.g. P18_02). 
- You should verify that the output looks ok, then you can set `generate_all` to True. 

## Running the code
**You should check g2 once in a while bc you might get kicked off a gpu. The intermediate progress will be saved!**

0. verify that you have all the proper settings for latest_config.yaml
    - stuff you might change: "llava_prompt_name", "generate_all", "generate_from_id", "prompt"
1. ssh into G2 and start/reattach your tmux session
2. Grab one gpu (either a 3090 or a6000. a6000 is kind of a overkill)
    - you can increase the time to be longer
```
srun --partition=gpu-interactive --gres=gpu:3090:1 --nodes=1 --ntasks=1 --mem=50000 --time=2:00:00 --pty bash
```
```
srun --partition=gpu-interactive --gres=gpu:a6000:1 --nodes=1 --ntasks=1 --mem=50000 --time=2:00:00 --pty bash
```
3. `conda activate llava`
4. cd into LLaVA folder (you terminal's pwd should say `/share/portal/<your_netid>/LLaVA`)
5. run `bash scripts/llava_as_api.sh`

# Details
## latest_settings.yml
- "llava_prompt_name": the version id of the prompt for gpt-4. This should change when you modify the "prompt" or the "query_template"
- "generate_all": set true if you want to generate results for all the videos that we want to generate. 
- "generate_from_id": a list of video id to generate result for. If generate_all is set to be True, this will be ignored
- "image_to_use": has to be one of ["start_image", "end_image"]. We should just use "start_image"! This is the image that LLaVA will look at for each timestep. 
- "prompt"
    - "system": system message to LLaVA. 
    - "query_template": a list of questions to ask LLaVA. <image> tag and <obj_list> must be in the first question. **(For v0, this should just be one question!)**
- "model_path": path to the LLaVA model (shouldn't need to change)
- "data_path": path to the raw data (shound't need to change on a regular basis)
- "overall_outputs_folder": path to the output folder (shound't need to change on a regular basis)
- "conv_mode": conversation setting for LLaVA (shouldn't need to change)