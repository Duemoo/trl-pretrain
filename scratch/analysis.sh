# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-105b_extended_5e-5_8_long.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs


# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name opt-1.3b_extended_1e-5_8_longgen.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/from-105-main

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name pythia_extended_1e-5_8_longgen.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/

# # ----------------------------------------------------------------------

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-105b_extended_5e-5_4_longgen_lr-adj.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-105b_extended_5e-5_16_longgen_lr-adj.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-105b_extended_5e-5_32_longgen_lr-adj.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-105b_extended_5e-5_64_longgen_lr-adj.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-1T_extended_5e-5_8_longgen.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/from-105-main

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-1.5T_extended_5e-5_8_longgen.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/from-105-main

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-2T_extended_5e-5_8_longgen.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/from-105-main


# # -----------------------------------------------------------------------

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-1.5T_extended_5e-5_8_long.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name tinyllama-2T_extended_5e-5_8_long.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs

# # python ppl_analysis.py \
# #     --mode measure_scores \
# #     --exp_name opt-1.3b_extended_1e-5_8_long.json \
# #     --text_log ref.json \
# #     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs


# # -------------------------------------------------------------

# python ppl_analysis.py \
#     --mode measure_scores \
#     --exp_name tinyllama-105b_extended_5e-5_4_longgen.json \
#     --text_log ref_4.json \
#     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs

# python ppl_analysis.py \
#     --mode measure_scores \
#     --exp_name tinyllama-105b_extended_5e-5_8_longgen.json \
#     --text_log ref.json \
#     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs

# python ppl_analysis.py \
#     --mode measure_scores \
#     --exp_name tinyllama-105b_extended_5e-5_16_longgen.json \
#     --text_log ref_16.json \
#     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs

# python ppl_analysis.py \
#     --mode measure_scores \
#     --exp_name tinyllama-105b_extended_5e-5_32_longgen.json \
#     --text_log ref_32.json \
#     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs

# python ppl_analysis.py \
#     --mode measure_scores \
#     --exp_name tinyllama-105b_extended_5e-5_64_longgen.json \
#     --text_log ref_64.json \
#     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs

# python ppl_analysis.py \
#     --mode measure_scores \
#     --exp_name tinyllama-105b_extended_5e-5_128_longgen.json \
#     --text_log ref_128.json \
#     --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs


# DIR='from-105-main'
DIR=''

# Define the array of interval values
INTERVALS=(300 500 1000 5000 10000 15000)

# Loop through each interval value
for INTERVAL in "${INTERVALS[@]}"
do
    python ppl_analysis.py \
        --mode measure_scores \
        --exp_name tinyllama-105b_extended_5e-5_128_longgen_lr-adj.json \
        --text_log ref_128.json \
        --base_dir /mnt/sda/hoyeon/trl-pretrain/results/logs/$DIR \
        --interval $INTERVAL
done