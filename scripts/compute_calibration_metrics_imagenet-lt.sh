
for MODEL in 'imagenet_lt_resnext50_crt'
do
for SEED in 2020 2021 2022 2023 2024
do

python compute_calibration_metrics_long_tail.py  \
    --model $MODEL \
    --distance_measure L2 \
    --data_dir_train "/home/miao/repo2022/yeeef/classifier-balancing/output/train" \
    --data_dir_val "/home/miao/repo2022/yeeef/classifier-balancing/output/val" \
    --random_seed $SEED  \
    --num_neighbors 10  

done
done