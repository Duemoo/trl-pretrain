lr="5e-5"
bsize="512"
exp_name="tinyllama-105b"
exp_name2="olmo-1b-500b"
exp_name3="olmo-1b-1400b"
exp_name4="olmo-1b-3100b"
postfix=""
dir=""

python ppl_analysis.py \
    --exp_name ${exp_name}_extended_${lr}_${bsize}_${postfix}.json \
    --text_log ref_${bsize}.json \
    --save_dir figs/olmo-1b_5e-5_64 \
    --save_dir figs/${exp_name}_${lr}_${bsize}_${postfix}


# python ppl_analysis.py \
    # --exp_name ${exp_name}_extended_${lr}_${bsize}_${postfix}.json ${exp_name2}_extended_${lr}_${bsize}_${postfix}.json ${exp_name3}_extended_${lr}_${bsize}_${postfix}.json ${exp_name4}_extended_${lr}_${bsize}_${postfix}.json \
    # --text_log ref_${bsize}.json \
    # --save_dir figs/olmo-1b_5e-5_64
    # --save_dir figs/${exp_name}_${lr}_${bsize}_${postfix}