for MODEL in 'vit_base_patch16_224' 'beit_large_patch16_224' 'mixer_b16_224' 'resnet50'
do
for SEED in 2020 2021 2022 2023 2024
do
python pytorch-image-models/precompute_intermediate_results.py \
    --data_dir "/home/miao/datasets/ImageNet" \
    --dataset "imagenet" \
    --model $MODEL \
    --output_dir "intermediate_output/imagenet" \
    --split 'val' 
done
done