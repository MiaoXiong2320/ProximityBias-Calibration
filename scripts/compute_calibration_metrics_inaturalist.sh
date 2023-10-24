for MODEL in 'resnet50_from_scratch_inat2021_val' 'resnet50_pretrain_imagenet_train_inat2021_val' 
do
for SEED in 2020 2021 2022 2023 2024
do
# "/home/miao/repo2022/yeeef/classifier-balancing/output/" 
python compute_calibration_metrics_long_tail_inaturalist.py  \
    --model $MODEL \
    --distance_measure L2 \
    --data_dir "/home/miao/repo2022/yeeef/newt/benchmark/output" \
    --random_seed $SEED  \
    --num_neighbors 10  

done
done