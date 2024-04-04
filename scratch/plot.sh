lr="1e-5"
bsize="16"
exp_name="olmo-7b-133b"
# exp_name="amber-36"
# exp_name2="amber-100"
# exp_name3="amber-200"
# exp_name4="amber-300"
# exp_name="tinyllama-105b"
# exp_name2="tinyllama-1T"
# exp_name3="tinyllama-2T"
# exp_name4="tinyllama-3T"
postfix=""
dir=""

python ppl_analysis.py \
    --exp_name ${exp_name}_extended_${lr}_${bsize}_${postfix}.json \
    --text_log precomputed_idx_${bsize}_121.json \
    --save_dir figs/${exp_name}_${lr}_${bsize}_${postfix}


# python ppl_analysis.py \
#     --exp_name ${exp_name}_extended_${lr}_${bsize}_${postfix}.json ${exp_name2}_extended_${lr}_${bsize}_${postfix}.json ${exp_name3}_extended_${lr}_${bsize}_${postfix}.json ${exp_name4}_extended_${lr}_${bsize}_${postfix}.json \
#     --text_log ref_${bsize}_121.json \
#     --save_dir figs/tinyllama_25e-6_64