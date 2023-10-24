# the parameters below such as INTENSITY/TRANSFORM does not matter for imagenet sketch inference
for MODEL in  'resnet50'
do
for INTENSITY in 1
do 
for TRANSFORM in 'gaussian_noise'
do
python imagenet_sketch_inference.py \
    "/home/miao/datasets/ImageNet-C/${TRANSFORM}" \
    --model $MODEL \
    --intensity $INTENSITY \
    --transform $TRANSFORM  

    
done
done
done