# Train
# accelerate launch --config_file config/1.yaml ./examples/scripts/inject.py \
#     --log_fpath results/logs/105b_inject_wo_resume.json \
#     --output_dir results/ckpts/105b_inject \
#     --model_name results/ckpts/105b_pre \
#     --devices 2 \
#     --max_steps 1

# accelerate launch --config_file config/1.yaml ./examples/scripts/inject.py \
#     --log_fpath results/logs/1T_inject.json \
#     --output_dir results/ckpts/1T_pre \
#     --model_name results/ckpts/1T_pre \
#     --devices 2 \
#     --resume \
#     --max_steps 101

# accelerate launch --config_file config/1.yaml ./examples/scripts/inject.py \
#     --log_fpath results/logs/1.5T_inject.json \
#     --output_dir results/ckpts/1.5T_pre \
#     --model_name results/ckpts/1.5T_pre \
#     --devices 2 \
#     --resume \
#     --max_steps 101

# accelerate launch --config_file config/1.yaml ./examples/scripts/inject.py \
#     --log_fpath results/logs/2T_inject.json \
#     --output_dir results/ckpts/2T_pre \
#     --model_name results/ckpts/2T_pre \
#     --devices 2 \
#     --resume \
#     --max_steps 101


CUDA_VISIBLE_DEVICES=7 python ./examples/scripts/inject.py \
    --log_fpath results/logs/test.json \
    --output_dir results/ckpts/2T_inject \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --max_steps 5 \
    --log_id 'test_log.json'