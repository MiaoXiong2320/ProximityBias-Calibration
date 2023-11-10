for MODEL in 'roberta-base'
do
python train_roberta_yahoo.py \
    --data_dir "intermediate_output/yahoo_answers_topics" \
    --dataset_name "yahoo_answers_topics" \
    --model $MODEL

python extract_roberta_yahoo.py \
    --data_dir "intermediate_output/yahoo_answers_topics" \
    --dataset_name "yahoo_answers_topics" \
    --model $MODEL 
done