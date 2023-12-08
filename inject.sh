# For evaluation purpose
CUDA_VISIBLE_DEVICES=0 python ./examples/scripts/inject.py \
    --log_fpath results/logs/105b_inject.json \
    --model_name TinyLlama/TinyLlama-1.1B-step-50K-105b \
    --learning_rate 0

CUDA_VISIBLE_DEVICES=1 python ./examples/scripts/inject.py \
    --log_fpath results/logs/1T_inject.json \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T \
    --learning_rate 0

CUDA_VISIBLE_DEVICES=2 python ./examples/scripts/inject.py \
    --log_fpath results/logs/1.5T_inject.json \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T \
    --learning_rate 0

CUDA_VISIBLE_DEVICES=3 python ./examples/scripts/inject.py \
    --log_fpath results/logs/2T_inject.json \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T \
    --learning_rate 0

# Train
CUDA_VISIBLE_DEVICES=4 python ./examples/scripts/inject.py \
    --log_fpath results/logs/105b_inject.json \
    --output_dir results/ckpts/105b_inject \
    --model_name TinyLlama/TinyLlama-1.1B-step-50K-105b

CUDA_VISIBLE_DEVICES=5 python ./examples/scripts/inject.py \
    --log_fpath results/logs/1T_inject.json \
    --output_dir results/ckpts/1T_inject \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T

CUDA_VISIBLE_DEVICES=6 python ./examples/scripts/inject.py \
    --log_fpath results/logs/1.5T_inject.json \
    --output_dir results/ckpts/1.5T_inject \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T

CUDA_VISIBLE_DEVICES=7 python ./examples/scripts/inject.py \
    --log_fpath results/logs/2T_inject.json \
    --output_dir results/ckpts/2T_inject 1\
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T