OUTPUT_DIR=model_output
EXP_NAME=qwen_main
mkdir ${OUTPUT_DIR}/${EXP_NAME}

accelerate launch --main_process_port=40067 --config_file pilot_llm/default_deepspeed.yaml pilot_llm/train_pilot_llm.py \
    --model_name_or_path="pilot_llm/weights/Qwen2-VL-2B-Instruct" \
    --reload_model_path="pilot_llm/weights/pretrained_Qwen2-VL-2B-Instruct" \
    --mission="prob" \
    --output_dir ${OUTPUT_DIR}/${EXP_NAME} \
    --mixed_precision "bf16" \
    --lora=True \
    --per_device_train_batch_size=2 \
    --checkpointing_steps=1000 \
    --max_train_steps=25000 \
    --checkpoints_total_limit=4 \
    2>&1 | tee ${OUTPUT_DIR}/${EXP_NAME}/train.log
