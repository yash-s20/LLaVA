#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="/share/portal/ys749/LLaVA/checkpoints/llava-llama-2-7b-chat-hf-lightning-merge"
# MODEL_VERSION="lmsys/vicuna-7b-v1.3"
PRETRAIN_NAME=llama-2-7b-chat
MODEL_NAME="$(basename $MODEL_VERSION)"
################## LLaMA-2 ##################

WANDB_MODE="offline" deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_train_v4-one-video.json \
    --image_folder "" \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/$PRETRAIN_NAME-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/$MODEL_NAME-finetune_lora_r2_debug \
    --num_train_epochs 200 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --lora_r 2
