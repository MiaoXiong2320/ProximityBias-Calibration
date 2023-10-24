for MODEL in  'resnet50'
do
for INTENSITY in 1
do 
for TRANSFORM in 'gaussian_noise'
do
python imagenet_c_inference.py \
    "/home/miao/datasets/ImageNet-C/${TRANSFORM}" \
    --model $MODEL \
    --intensity $INTENSITY \
    --transform $TRANSFORM  
    
done
done
done