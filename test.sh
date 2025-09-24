CUDA_VISIBLE_DEVICES=0,1

# MODEL='llava'
MODEL='llava_onevision' 

# python -m debugpy --connect 5679 -m accelerate.commands.launch \
python -m accelerate.commands.launch \
    --num_processes=2 \
    --main_process_port 12346 \
    -m lmms_eval \
    --model $MODEL \
    --model_args pretrained="/home/yilin/MM-RLHF/output/DPO/llava-onevision-qwen2-0.5b-ov_mmrlhf-w0.1-beta0.1-epoch7/checkpoint-14000" \
    --tasks amber \
    --batch_size 1 \
    --output_path "/home/yilin/mmrlhf-eval/output/" \
    --wandb_log_samples \
    --wandb_args "project=lmms-eval,job_type=eval" \
    