"""
This code is used to train the RoBERTa classifier on Yahoo Answer Topics dataset.
"""

#%%
from argparse import ArgumentParser
import os, pdb
import os.path as osp
import random
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import torch


def check_manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed) # for cpu
    torch.cuda.manual_seed(seed) # for single GPU
    torch.cuda.manual_seed_all(seed) # for all GPUs
    torch.backends.cudnn.benchmark = False
    print("Using seed: {seed}".format(seed=seed))



#%%
parser = ArgumentParser()

parser.add_argument("--dataset_name", type=str, default="yahoo_answers_topics") 
parser.add_argument("--data_dir", type=str, default="intermediate_output/yahoo_answers_topics/")
parser.add_argument("--normalize", type=bool, default=True)
parser.add_argument("--num_neighbors", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=2022)
parser.add_argument("--distance_measure", type=str, default="L2") # L2, cosine, IVFFlat, IVFPQ

parser.add_argument('--model', 
                    default='roberta-base',
)


args = parser.parse_args()

check_manual_seed(args.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
################ LOAD DATASET ############################
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import datasets
import evaluate as hg_eval

# Load Yahoo Answers dataset
dataset = datasets.load_dataset('yahoo_answers_topics')
dataset['train'] = dataset['train'].rename_column('topic', 'label')
dataset['test'] = dataset['test'].rename_column('topic', 'label')

# Load RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=10)

# Tokenize and preprocess the dataset
def tokenize_dataset(examples):
    return tokenizer(examples['question_title'], padding="max_length", truncation=True)
tokenized_dataset = dataset.map(tokenize_dataset, batched=True)

train_size = len(dataset["train"])
val_size = int(train_size * 0.2)
val_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(0, val_size))
train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(val_size, train_size))

# TODO
# val_dataset = dataset["train"].shuffle(seed=42).select(range(0, val_size))
# train_dataset = dataset["train"].shuffle(seed=42).select(range(val_size, train_size))

metric = hg_eval.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Set up the Trainer and TrainingArguments
training_args = TrainingArguments(
    output_dir='train_model_output',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='train_model_output/logs',
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    # train_dataset=val_dataset,
    
    eval_dataset=val_dataset,  
    compute_metrics=compute_metrics,
    # tokenizer=tokenizer,
    # data_collator=data_collator 
    # data_collator=lambda data: {'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
    #                             'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
    #                             'labels': torch.tensor([f['topic'] for f in data])}
)

# Train the model
trainer.train()
trainer.save_model("train_model_output")
print(trainer.evaluate(val_dataset))
