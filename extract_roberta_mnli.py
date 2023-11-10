"""
This code is used to extract the intermediate results of RoBERTa on MNLI dataset.
"""

#%%
import pandas as pd
from argparse import ArgumentParser
import os, pdb
import os.path as osp
import random
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import torch
import tqdm


def check_manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed) # for cpu
    torch.cuda.manual_seed(seed) # for single GPU
    torch.cuda.manual_seed_all(seed) # for all GPUs
    torch.backends.cudnn.benchmark = False
    print("Using seed: {seed}".format(seed=seed))



#%%
parser = ArgumentParser()

parser.add_argument("--dataset_name", type=str, default="mnli") 
parser.add_argument("--data_dir", type=str, default="intermediate_output/mnli/")
parser.add_argument("--normalize", type=bool, default=True)
parser.add_argument("--num_neighbors", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=2022)
parser.add_argument("--distance_measure", type=str, default="L2") # L2, cosine, IVFFlat, IVFPQ

parser.add_argument('--model', 
                    default='roberta-base',
)


args = parser.parse_args()

check_manual_seed(args.random_seed)

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
################ LOAD DATASET ############################
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaModel
import datasets
import evaluate as hg_eval
from torch.utils.data import Dataset, DataLoader
from torch import nn

N_LABELS = 3
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.tokenized_dataset[idx]["input_ids"]),
                "attention_mask": torch.tensor(self.tokenized_dataset[idx]["attention_mask"]),
                "label": torch.tensor(self.tokenized_dataset[idx]["label"])}

# Load Yahoo Answers dataset
dataset = datasets.load_dataset('glue', 'mnli')

# Load RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model)

# model = RobertaModel.from_pretrained('train_model_output', num_labels=10).cuda()
# TODO seems no need to train; the accuracy is already 86%
model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained('WillHeld/roberta-base-mnli', num_labels=N_LABELS).cuda())
base_model = nn.DataParallel(RobertaModel.from_pretrained('WillHeld/roberta-base-mnli', num_labels=N_LABELS).cuda())


# Tokenize and preprocess the dataset
def tokenize_dataset(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True)
tokenized_dataset = dataset.map(tokenize_dataset, batched=True)

val_matched_dataset = tokenized_dataset["validation_matched"]
val_mismachted_dataset = tokenized_dataset["validation_mismatched"]

def extract_everything(dataset):
    batch_size = 512  # Choose an appropriate batch size for your problem
    dataloader = DataLoader(CustomDataset(dataset), batch_size=batch_size, shuffle=False, num_workers=56, pin_memory=True)


    model.eval()
    base_model.eval()
    logits_list = []
    confs_list = []
    preds_list = []
    embeds_list = []
    labels = np.array([dataset[idx]['label'] for idx in range(len(dataset))])

    for batch in tqdm.tqdm(dataloader):
        # input_ids = torch.tensor(batch["input_ids"]).cuda().unsqueeze(0)
        # attention_mask = torch.tensor(batch["attention_mask"]).cuda().unsqueeze(0)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)

        with torch.no_grad():
            outputs = base_model(input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            
            outputs = model(input_ids, attention_mask=attention_mask)
            # outputs = model(inputs_embeds=batch_embeddings)
            batch_logits = outputs.logits
            batch_confs = nn.functional.softmax(batch_logits, dim=-1)
            batch_predictions = torch.argmax(batch_logits, dim=-1)
            
            embeds_list.append(batch_embeddings.cpu().numpy())
            logits_list.append(batch_logits.cpu().numpy())
            preds_list.append(batch_predictions.cpu().numpy())
            confs_list.append(batch_confs.cpu().numpy())
            
        # logits.extend(batch_logits.tolist())
        # predictions.extend(batch_predictions.tolist())
    metric = hg_eval.load("accuracy")

    logits = np.concatenate(logits_list) # logits
    preds = np.concatenate(preds_list) # preds
    zs = np.concatenate(embeds_list) # zs
    confs = np.concatenate(confs_list) # confs
    print(logits.shape, preds.shape, zs.shape, confs.shape, labels.shape)
    # {'accuracy': 0.8656427990235964}
    print(f"acc: {metric.compute(predictions=preds, references=labels)}")
    preds = np.eye(N_LABELS)[preds]   
    ys = np.eye(N_LABELS)[labels] # ys
    print(logits.shape, preds.shape, zs.shape, confs.shape, ys.shape)

    return logits, preds, zs, confs, ys

save_dir = f"intermediate_output/{args.dataset_name}/"
os.makedirs(save_dir, exist_ok=True)

logits, preds, zs, confs, ys = extract_everything(val_matched_dataset)
np.savez(os.path.join(save_dir, "mnli_val_matched.npz"), logits=logits, preds=preds, zs=zs, confs=confs, ys=ys)
logits, preds, zs, confs, ys = extract_everything(val_mismachted_dataset)
np.savez(os.path.join(save_dir, "mnli_val_mismatched.npz"), logits=logits, preds=preds, zs=zs, confs=confs, ys=ys)

