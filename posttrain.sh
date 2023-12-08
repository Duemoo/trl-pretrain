accelerate launch --num_processes 8 ./examples/scripts/sft.py \
    --log_fpath results/logs/1T_post.json \
    --output_dir results/ckpts/1T_post \
    --model_name results/ckpts/1T_inject \
    --max_steps 500 \
    --log_with wandb \
    --devices 8


