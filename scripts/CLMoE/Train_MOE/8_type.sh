################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################


################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

deepspeed --include localhost:0,1 --master_port 29600 llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero3_offload.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --expert_num 4 \
    --model_name_or_path /your/catalogue/CLMoE/checkpoint/vicuna-7b-v1.5 \
    --previous_task_model_path ./checkpoints/CL4VQA/color/llava-1.5-7b-lora \
    --version $PROMPT_VERSION \
    --data_path /your/catalogue/CLMoE/CL4VQA/train/train_q_type.json \
    --image_folder /your/catalogue/CLMoE/data/ \
    --vision_tower /your/catalogue/CLMoE/checkpoint/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/CL4VQA/Only_Pretrain_1.5_MOE_2/type/llava-1.5-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --task type