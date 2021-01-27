#! usr/bin/env python
# -*- coding : utf-8 -*-

import pandas as pd
pd.set_option('max_colwidth', 400)
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, \
    classification_report
from glob import glob
import os
import time
import codecs
from utils import load_tab_data
from args import get_args
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging
logging.basicConfig(filename='../logs/bert_models_results.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  datefmt='%H:%M:%S', level=logging.DEBUG)
# specify GPU
device = torch.device("cuda")

args = get_args()

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        self.fc2 = nn.Linear(512, 256)

        # dense layer 2 (Output layer)
        self.fc3 = nn.Linear(256, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        x = self.fc3(x)

        # apply softmax activation
        x = self.softmax(x)

        return x


# function to train the model
def train(train_dataloader, model, cross_entropy, optimizer):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate(val_dataloader, model, cross_entropy):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def bert_network(X_train, y_train, X_dev, y_dev, epochs=10):
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    for param in bert.parameters():
        param.requires_grad = False  # freeze all the parameters
    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(bert)


    # push the model to GPU
    model = model.to(device)

    # optimizer from hugging face transformers


    # define the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=1e-5)


    best_valid_loss = float('inf')

    # define a batch size
    batch_size = 32

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        X_train.tolist(),
        max_length=100,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        X_dev.tolist(),
        max_length=100,
        pad_to_max_length=True,
        truncation=True
    )



    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(y_train.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(y_dev.tolist())




    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # compute the class weights
    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

    print("Class Weights:", class_weights)

    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    if not args.only_test:
        # for each epoch
        a = time.time()
        for epoch in range(epochs):

            print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

            # train model
            train_loss, _ = train(train_dataloader, model, cross_entropy, optimizer)

            # evaluate model
            valid_loss, _ = evaluate(val_dataloader, model, cross_entropy)

            # save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), '../models/saved_weights_bert.pt')

            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')
        fit_time = time.time() - a

        logging.info("=================================START====================================\n")
        logging.info("==============================TRAINING====================================")
        logging.info("dataset name: %s", args.train_data)
        logging.info("fit_time: %s", fit_time)
        logging.info("\n")

    logging.info("==========================TESTING==========================================")

    path = '../models/saved_weights_bert.pt'
    model.load_state_dict(torch.load(path))

    for test_file in glob(os.path.join(args.test_path, '*test_data*obfuscated.txt')):
        X_test_o, y_test = load_tab_data(filename=test_file, preprocessed=True, test_file=True)

        tokens_test = tokenizer.batch_encode_plus(
            X_test_o.tolist(),
            max_length=100,
            pad_to_max_length=True,
            truncation=True
        )
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        test_y = torch.tensor(y_test.tolist())
        a = time.time()
        with torch.no_grad():
            preds = model(test_seq.to(device), test_mask.to(device))
            preds = preds.detach().cpu().numpy()
        score_time = time.time() - a
        y_test = y_test
        y_preds = np.argmax(preds, axis=1)
        result_file = os.path.join(args.evaluate_label_path, os.path.basename(test_file).replace('.txt', '') + '_' +
                                   os.path.basename(args.train_data) + '_bert_network_predictions.txt')

        with codecs.open(result_file, 'w', 'utf-8') as result_obj:
            for i, val in enumerate(X_test_o):
                result_obj.write(str(val) + '\t' + str(y_preds[i]) + '\n')

        y_pred = []
        for i in y_preds:
            if i >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        logging.info("Test file : %s", result_file)
        logging.info("F1 Score: %s", f1_score(y_test, y_pred))
        logging.info("Score_time : %s", score_time)
        logging.info("Confusion Matrix : %s", confusion_matrix(y_test, y_pred))
        target_names = ['Non-hate', 'hate']
        logging.info("Classification Report : %s", classification_report(y_test, y_pred, target_names=target_names))

    logging.info("==================================END======================================\n\n")



def main():
    X_train_o, y_train = load_tab_data(filename=args.train_data, preprocessed=True)
    X_dev_o, y_dev = load_tab_data(filename=args.dev_data, preprocessed=True)
    bert_network(X_train_o, y_train, X_dev_o, y_dev, epochs=10)



if __name__ == '__main__':
    main()



