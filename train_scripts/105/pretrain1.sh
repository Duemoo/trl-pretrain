base_dir=/mnt/nas/hoyeon/trl-pretrain
gpu=2
lr=5e-5
bsize=64
micro_bsize=1
postfix=test
base_model=TinyLlama/TinyLlama-1.1B-step-50K-105b
model=tinyllama-105b
revision=-1
is_llama=true

accelerate launch --config_file config/${gpu}.yaml /mnt/nas/hoyeon/trl-pretrain/examples/scripts/sft.py \
    --log_fpath ${base_dir}/results/logs/${model}_pre_${lr}_${bsize}_${postfix}.json \
    --output_dir ${base_dir}/results/ckpts/${model}_${lr}_${bsize}_${postfix} \
    --model_name ${base_model} \
    --max_steps 100 \
    --devices 1 \
    --learning_rate ${lr} \
    --micro_batch_size ${micro_bsize} \
    --global_batch_size ${bsize} \
    --revision ${revision} \
    --is_llama ${is_llama}


accelerate launch --config_file config/${gpu}.yaml /mnt/nas/hoyeon/trl-pretrain/examples/scripts/sft.py \
    --log_fpath ${base_dir}/results/logs/${model}_post_${lr}_${bsize}_${postfix}.json \
    --output_dir ${base_dir}/results/ckpts/${model}_${lr}_${bsize}_${postfix} \
    --model_name ${base_dir}/results/ckpts/${model}_${lr}_${bsize}_${postfix} \
    --max_steps 500 \
    --devices 1 \
    --learning_rate ${lr} \
    --log_id ${base_dir}/results/logs/${model}_post_text_${lr}_${bsize}_${postfix}.json \
    --mixed_train true \
    --resume true \
    --micro_batch_size ${micro_bsize} \
    --global_batch_size ${bsize} \
    --revision ${revision} \
    --is_llama ${is_llama}


accelerate launch --config_file config/${gpu}.yaml /mnt/nas/hoyeon/trl-pretrain/examples/scripts/sft.py \
    --log_fpath ${base_dir}/results/logs/${model}_postpost_${lr}_${bsize}_${postfix}.json \
    --output_dir ${base_dir}/results/ckpts/${model}_${lr}_${bsize}_${postfix} \
    --model_name ${base_dir}/results/ckpts/${model}_${lr}_${bsize}_${postfix} \
    --max_steps 20000 \
    --devices 1 \
    --learning_rate ${lr} \
    --log_id ${base_dir}/results/logs/${model}_postpost_text_${lr}_${bsize}_${postfix}.json \
    --resume true \
    --micro_batch_size ${micro_bsize} \
    --global_batch_size ${bsize} \
    --revision ${revision} \
    --is_llama ${is_llama} \
    --fast_eval True
