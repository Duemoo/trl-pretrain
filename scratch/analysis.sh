DIR=''

lr="5e-5"
bsize="64"
exp_name="olmo-1b-3100b"
postfix=""
dir=""

# Define the array of interval values
INTERVALS=(300 500 1000 3000 5000 10000)

# Loop through each interval value
for INTERVAL in "${INTERVALS[@]}"
do
    python ppl_analysis.py \
        --mode measure_scores \
        --exp_name ${exp_name}_extended_${lr}_${bsize}_${postfix}.json \
        --text_log ref_${bsize}.json \
        --base_dir /mnt/nas/hoyeon/trl-pretrain/results/logs/$DIR \
        --interval $INTERVAL
done