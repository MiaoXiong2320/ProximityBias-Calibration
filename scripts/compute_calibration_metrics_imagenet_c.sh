
for MODEL in  'resnet50' 
do
for SEED in  2022 
do
for CORRUPTION in 'brightness' 'elastic_transform' 'frost' 'gaussian_blur' 'glass_blur' 'impulse_noise' 'jpeg_compression' 'motion_blur' 'pixelate' 'shot_noise' 'snow' 'spatter' 'speckle_noise' 'zoom_blur' "gaussian_noise" 
do 
for INTENSITY in 1
do
python compute_calibration_metrics_distribution_shift.py  \
    --model $MODEL \
    --distance_measure L2 \
    --random_seed $SEED  \
    --num_neighbors 10  \
    --data_dir_val "intermediate_output/imagenet/" \
    --data_dir_train "None"  \
    --data_dir_test "intermediate_output/imagenet_C"  \
    --corruption $CORRUPTION \
    --intensity $INTENSITY
done
done
done
done