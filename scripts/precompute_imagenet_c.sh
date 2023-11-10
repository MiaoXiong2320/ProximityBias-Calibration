for MODEL in  'resnet50'
do
for INTENSITY in 1
do 
for TRANSFORM in 'brightness' 'elastic_transform' 'frost' 'gaussian_blur' 'glass_blur' 'impulse_noise' 'jpeg_compression' 'motion_blur' 'pixelate' 'shot_noise' 'snow' 'spatter' 'speckle_noise' 'zoom_blur'
do

python pytorch-image-models/imagenet_c_inference.py \
    "/home/miao/datasets/ImageNet-C/${TRANSFORM}" \
    --model $MODEL \
    --intensity $INTENSITY \
    --transform $TRANSFORM  
done
done
done