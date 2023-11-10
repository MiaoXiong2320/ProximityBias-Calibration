for DATASET_NAME in "mnli_matched" "mnli_mismatched"
do
for SEED in 2022
do
python compute_calibration_metrics_nlp.py  \
    --dataset_name $DATASET_NAME \
    --distance_measure L2 \
    --random_seed $SEED  \
    --num_neighbors 10  
done
done