for MODEL in  'resnet50' 
do
for SEED in  2022 
do
for CORRUPTION in 'none'
do 
for INTENSITY in 0
do
python compute_calibration_metrics_distribution_shift.py  \
    --model $MODEL \
    --dataset_name "imagenet_sketch" \
    --distance_measure L2 \
    --random_seed $SEED  \
    --num_neighbors 10  \
    --data_dir_val "pytorch_image_models/intermediate_output/imagenet/" \
    --data_dir_train "pytorch_image_models/intermediate_output/imagenet_train"  \
    --data_dir_test "pytorch_image_models/intermediate_output/imagenet_sketch"  \
    --corruption $CORRUPTION \
    --intensity $INTENSITY
    

done
done
done
done