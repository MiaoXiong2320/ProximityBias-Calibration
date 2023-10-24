for MODEL in 'resnet18' 'vit_base_patch16_224_sam' 'xcit_medium_24_p8_224' 'deit3_large_patch16_384_in21ft1k' 'volo_d5_224'
do

for SEED in 2020 2021 2022 2023 2024
do

python compute_calibration_metrics.py  \
    --model $MODEL \
    --distance_measure L2 \
    --data_dir "pytorch_image_models/intermediate_output/imagenet/" \
    --random_seed $SEED  \
    --num_neighbors 10  

done
done