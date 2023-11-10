for MODEL in 'roberta-base'
do
python train_roberta_mnli.py \
    --data_dir "intermediate_output/mnli" \
    --dataset_name "mnli" \
    --model $MODEL

python extract_roberta_mnli.py \
    --data_dir "intermediate_output/mnli" \
    --dataset_name "mnli" \
    --model $MODEL 

python 
done