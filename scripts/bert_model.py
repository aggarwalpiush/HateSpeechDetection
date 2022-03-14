#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from args import get_args
import logging
logging.basicConfig(filename='../logs/bert_models_results.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  datefmt='%H:%M:%S', level=logging.DEBUG)

args = get_args()


df = pd.read_csv(args.train_data,
                   sep='\t',
                   header=None, 
                   names=['article', 'label'],
                   encoding='ISO-8859-1')
df1 = pd.read_csv(args.dev_data,
                   sep='\t',
                   header=None, 
                   names=['article', 'label'],
                   encoding='ISO-8859-1')




texts = df['article'].values
labels = df['label'].values




texts1 = df1['article'].values
labels1 = df1['label'].values










text_lengths = [len(texts[i].split()) for i in range(len(texts))]



text_lengths1 = [len(texts1[i].split()) for i in range(len(texts1))]






from transformers import DistilBertTokenizer
from transformers import AutoModel, AutoTokenizer

if args.use_de_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-dbmdz-uncased")
    #model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
    #tokenizer = DistilBertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=True)
else:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)




text_ids = [tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in texts]




text_ids1 = [tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in texts1]







text_ids_lengths = [len(text_ids[i]) for i in range(len(text_ids))]



text_ids_lengths1 = [len(text_ids1[i]) for i in range(len(text_ids1))]





att_masks = []
for ids in text_ids:
    masks = [int(id > 0) for id in ids]
    att_masks.append(masks)


att_masks1 = []
for ids in text_ids1:
    masks = [int(id > 0) for id in ids]
    att_masks1.append(masks)




    

train_x = text_ids
train_y = labels
train_m = att_masks

val_x = text_ids1
val_y = labels1
val_m = att_masks1





# In[129]:


import torch

train_x = torch.tensor(train_x)

val_x = torch.tensor(val_x)
train_y = torch.tensor(train_y)

val_y = torch.tensor(val_y)
train_m = torch.tensor(train_m)

val_m = torch.tensor(val_m)



import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 32

train_data = TensorDataset(train_x, train_m, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_x, val_m, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)



from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM
num_labels = len(set(labels))

if args.use_de_tokenizer:
    #model = DistilBertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=num_labels,
    #                                                        output_attentions=False, output_hidden_states=False)
    model = AutoModelForMaskedLM.from_pretrained("bert-base-german-dbmdz-uncased")
else:
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



learning_rate = 1e-5
adam_epsilon = 1e-8

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)


# In[136]:


from transformers import get_linear_schedule_with_warmup

num_epochs = 3
total_steps = len(train_dataloader) * num_epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)




import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

import numpy as np
import random

seed_val = 111

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)




train_losses = []
val_losses = []
num_mb_train = len(train_dataloader)
num_mb_val = len(val_dataloader)

if num_mb_val == 0:
    num_mb_val = 1



if not args.only_test:
    # for each epoch
    a = time.time()
    for n in range(num_epochs):
        train_loss = 0
        val_loss = 0
        start_time = time.time()

        for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()

            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)

            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)

            loss = outputs[0]
            # loss = model_loss(outputs[1], mb_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.data / num_mb_train

        print("\nTrain loss after itaration %i: %f" % (n + 1, train_loss))
        train_losses.append(train_loss.cpu())

        with torch.no_grad():
            model.eval()

            for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
                mb_x = mb_x.to(device)
                mb_m = mb_m.to(device)
                mb_y = mb_y.to(device)

                outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)

                loss = outputs[0]
                # loss = model_loss(outputs[1], mb_y)

                val_loss += loss.data / num_mb_val

            print("Validation loss after itaration %i: %f" % (n + 1, val_loss))
            val_losses.append(val_loss.cpu())

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Time: {epoch_mins}m {epoch_secs}s')
    torch.save(model.state_dict(), '../models/saved_weights_bert.pt')
    fit_time = time.time() - a

    logging.info("=================================START====================================\n")
    logging.info("==============================TRAINING====================================")
    logging.info("dataset name: %s", args.train_data)
    logging.info("fit_time: %s", fit_time)
    logging.info("\n")

logging.info("==========================TESTING==========================================")

path = '../models/davidson/saved_weights_bert.pt'
model.load_state_dict(torch.load(path))

from glob import glob
import os
import codecs
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, \
    classification_report

for test_file in glob(os.path.join(args.test_path, '*test_data*obfuscated.txt')):
    test_tile1 = time.time()
    df2 = pd.read_csv(test_file,
                      sep='\t',
                      header=None,
                      names=['article', 'label'],
                      encoding='ISO-8859-1')

    texts2 = df2['article'].values
    labels2 = df2['label'].values

    text_lengths2 = [len(texts2[i].split()) for i in range(len(texts2))]
    text_ids2 = [tokenizer.encode(text, max_length=300, pad_to_max_length=True) for text in texts2]
    text_ids_lengths2 = [len(text_ids2[i]) for i in range(len(text_ids2))]

    att_masks2 = []
    for ids in text_ids2:
        masks = [int(id > 0) for id in ids]
        att_masks2.append(masks)
    test_x = text_ids2
    test_y = labels2
    test_m = att_masks2

    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)
    test_m = torch.tensor(test_m)

    test_data = TensorDataset(test_x, test_m)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    outputs = []
    with torch.no_grad():
        model.eval()
        for k, (mb_x, mb_m) in enumerate(test_dataloader):
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            output = model(mb_x, attention_mask=mb_m)
            outputs.append(output[0].to('cpu'))

    outputs = torch.cat(outputs)

    _, predicted_values = torch.max(outputs, 1)
    predicted_values = predicted_values.numpy()
    true_values = test_y.numpy()
    result_file = os.path.join(args.evaluate_label_path, os.path.basename(test_file).replace('.txt', '') + '_' +
                               os.path.basename(args.train_data) + '_bert_network_predictions.txt')

    with codecs.open(result_file, 'w', 'utf-8') as result_obj:
        for i, val in enumerate(texts2):
            result_obj.write(str(val) + '\t' + str(predicted_values[i]) + '\n')

    y_pred = []
    for i in predicted_values:
        if i >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    score_time = time.time() - test_tile1
    logging.info("Test file : %s", result_file)
    logging.info("F1 Score: %s", f1_score(true_values, y_pred))
    logging.info("Score_time : %s", score_time)
    logging.info("Confusion Matrix : %s", confusion_matrix(true_values, y_pred))
    target_names = ['Non-hate', 'hate']
    logging.info("Classification Report : %s", classification_report(true_values, y_pred, target_names=target_names))

logging.info("==================================END======================================\n\n")









