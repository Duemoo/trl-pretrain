DIR=''

lr="1e-6"
bsize="64"
exp_name="olmo-1b-126b"
postfix=""
dir=""

# Define the array of interval values
# INTERVALS=(100 300 500 1000 3000 5000)
INTERVALS=(100)

# Loop through each interval value
for INTERVAL in "${INTERVALS[@]}"
do
    python ppl_analysis.py \
        --mode measure_scores \
        --exp_name ${exp_name}_extended_${lr}_${bsize}_${postfix}.json \
        --text_log precomputed_idx_${bsize}_121.json \
        --base_dir /mnt/nas/hoyeon/trl-pretrain/ \
        --interval $INTERVAL
done