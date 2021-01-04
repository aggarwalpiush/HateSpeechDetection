#! usr/bin/env python
# -*- coding : utf-8 -*-

import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
pd.set_option('max_colwidth', 400)
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, Bidirectional, Dense, \
    LSTM, Conv1D, MaxPooling1D, Dropout, concatenate, Flatten, add
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.engine import Layer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import Input, Model
from keras.optimizers import Adam
from keras.models import Sequential, clone_model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import time
from utils import load_tab_data
from args import get_args
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
import wget
import zipfile
import logging
logging.basicConfig(filename='../logs/network_models_results.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  datefmt='%H:%M:%S', level=logging.DEBUG)
# specify GPU
device = torch.device("cuda")


args = get_args()





class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'supports_masking': self.supports_masking,
            'init': self.init,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias,
            'step_dim': self.step_dim,
            'features_dim': self.features_dim
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


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



strategy = tf.distribute.MirroredStrategy()


def cnn_attention_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"best_model_cnn_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    with strategy.scope():
        inp = Input(shape=(max_len,))
        x = Embedding(max_features + 1, embed_size, weights=[embedding_matrix], trainable=False)(inp)
        x1 = SpatialDropout1D(spatial_dr)(x)
        att = Attention(max_len)(x1)
        x = Conv1D(conv_size, 2, activation='relu', padding='same')(x1)
        x = MaxPooling1D(5, padding='same')(x)
        x = Conv1D(conv_size, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        x = Flatten()(x)
        x = concatenate([x, att])
        x = Dropout(dr)(Dense(dense_units, activation='relu')(x))
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    with strategy.scope():
        model2 = Model(inputs=inp, outputs=x)
        model2.load_weights(file_path)
        model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def cnn_lstm_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"best_model_cnn_lstm_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    model = Sequential()
    model.add(Embedding(max_features + 1, embed_size * 2, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Conv1D(200, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(100))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(1, activation='sigmoid'))
    model2 = model
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def lstm_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"best_model_lstm_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape = (max_len,),name='main_input')
    glove_Embed = (Embedding(max_features + 1, embed_size * 2, weights=[embedding_matrix], trainable=False))(main_input)
    y = LSTM(300)(glove_Embed)
    y = Dense(200, activation='relu')(y)
    y = Dropout(rate=0.15)(y)
    z = Dense(100, activation='relu')(y)
    output_lay = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[main_input], outputs=[output_lay])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model2 = Model(inputs=[main_input], outputs=[output_lay])
    model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def cnn_deep_lstm(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"best_model_cnn_deep_lstm.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    glove_Embed = (Embedding(max_features + 1, embed_size * 2, input_length=max_len, weights=[embedding_matrix], trainable=False))(main_input)

    x0 = Conv1D(128, 10, activation='relu')(glove_Embed)
    x1 = Conv1D(64, 5, activation='relu')(x0)
    x2 = Conv1D(32, 4, activation='relu')(x1)
    x3 = Conv1D(16, 3, activation='relu')(x2)
    x4 = Conv1D(8, 5, activation='relu')(x3)
    x = MaxPooling1D(pool_size=3)(x4)
    x = Dropout(rate=0.25)(x)
    x = LSTM(100)(x)

    p = MaxPooling1D(pool_size=10)(x0)
    p = Dropout(rate=0.15)(p)
    p = LSTM(100)(p)

    o = MaxPooling1D(pool_size=8)(x1)
    o = Dropout(rate=0.15)(o)
    o = LSTM(100)(o)

    i = MaxPooling1D(pool_size=6)(x2)
    i = Dropout(rate=0.15)(i)
    i = LSTM(100)(i)

    r = MaxPooling1D(pool_size=4)(x3)
    r = Dropout(rate=0.15)(r)
    r = LSTM(100)(r)

    t = MaxPooling1D(pool_size=3)(x4)
    t = Dropout(rate=0.15)(t)
    t = LSTM(100)(t)

    y = LSTM(500)(glove_Embed)
    y = Dense(250,activation='relu')(y)
    y = Dropout(rate=0.15)(y)

    z = concatenate([x, p, o, i, r, t, y])

    z = Dense(400,activation='relu')(z)
    z = Dropout(0.15)(z)
    z = Dense(200,activation='relu')(z)
    z = Dense(100,activation='relu')(z)
    z = Dropout(0.15)(z)
    z = Dense(50,activation='relu')(z)
    output_lay = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[main_input], outputs=[output_lay])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model2 = Model(inputs=[main_input], outputs=[output_lay])
    model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def bilstm_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"best_model_bilstm_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    x = (Embedding(max_features + 1, embed_size*2, input_length=max_len, weights=[embedding_matrix], trainable=False))(main_input)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = Dense(1024, activation='relu')(hidden)
    hidden = Dense(512, activation='relu')(hidden)
    output_lay = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=[main_input], outputs=[output_lay])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model2 = Model(inputs=[main_input], outputs=[output_lay])
    model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def bilstm_attention_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"best_model_bilstm_attention_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    with strategy.scope():
        main_input = Input(shape=(max_len,), name='main_input')
        x = (Embedding(max_features + 1, embed_size*2, input_length=max_len, weights=[embedding_matrix], trainable=False))(main_input)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        hidden = concatenate([
            Attention(max_len)(x),
            GlobalMaxPooling1D()(x),
        ])
        hidden = Dense(1024, activation='relu')(hidden)
        hidden = Dense(512, activation='relu')(hidden)
        output_lay = Dense(1, activation='sigmoid')(hidden)
        model = Model(inputs=[main_input], outputs=[output_lay])
        model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
        model2 = Model(inputs=[main_input], outputs=[output_lay])
    model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    with strategy.scope():
        model2.load_weights(file_path)
        model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def bert_network(X_train, y_train, X_dev, y_dev, X_test, y_test, epochs=10):
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

    tokens_test = tokenizer.batch_encode_plus(
        X_test.tolist(),
        max_length=100,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the test set
    # tokens_test = tokenizer.batch_encode_plus(
    #   test_text.tolist(),
    #  max_length = 100,
    # pad_to_max_length=True,
    # truncation=True
    # )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(y_train.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(y_dev.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(y_test.tolist())

    # test_seq = torch.tensor(tokens_test['input_ids'])
    # test_mask = torch.tensor(tokens_test['attention_mask'])
    # test_y = torch.tensor(test_labels.tolist())

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

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train(train_dataloader, model, cross_entropy)

        # evaluate model
        valid_loss, _ = evaluate(val_dataloader, model, cross_entropy, optimizer)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights_bert.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    path = 'saved_weights_bert.pt'
    model.load_state_dict(torch.load(path))


    a = time.time()
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    pred_time = time.time() -a
    y_test = y_test
    y_preds = np.argmax(preds, axis=1)
    f1 = f1_score(y_test, y_preds, average='macro')


    return f1, pred_time







def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def build_matrix(embedding_path, tk, max_features):
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding = "utf-8"))

    word_index = tk.word_index
    nb_words = max_features
    embedding_matrix = np.zeros((nb_words + 1, 300))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_embedding_matrix(embed, tk, max_features):
    if embed == 'fasttext':
        if not os.path.exists('../embeddings/crawl-300d-2M.vec'):
            wget.download('https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip', '../embeddings')
            with zipfile.ZipFile('../embeddings/crawl-300d-2M.vec.zip', "r") as zip_ref:
                zip_ref.extractall()
        return build_matrix('../embeddings/crawl-300d-2M.vec', tk, max_features)
    elif embed == 'glove':
        if not os.path.exists('../embeddings/glove.42B.300d.txt'):
            wget.download('http://nlp.stanford.edu/data/glove.42B.300d.zip',
                          '../embeddings')
            with zipfile.ZipFile('../embeddings/glove.42B.300d.zip', "r") as zip_ref:
                zip_ref.extractall()
        return build_matrix('../embeddings/glove.42B.300d.txt', tk, max_features)
    else:
        return np.concatenate([build_matrix('../embeddings/crawl-300d-2M.vec', tk, max_features),
                               build_matrix('../embeddings/glove.42B.300d.zip', tk, max_features)], axis=-1)


def network_pred_f1(model, X_test, y_test):
    y_preds = []
    a = time.time()
    for i in model.predict(X_test):
        if i[0] >= 0.5:
            y_preds.append(1)
        else:
            y_preds.append(0)
    score_time = time.time() - a
    return f1_score(y_test, y_preds), score_time


def main():
    max_features = args.max_features
    max_len = args.max_len
    embed_size = args.embed_size
    X_train, y_train = load_tab_data(filename=args.train_data, preprocessed=True)

    X_dev, y_dev = load_tab_data(filename=args.dev_data, preprocessed=True)

    X_test, y_test = load_tab_data(filename=args.test_data, preprocessed=True)
    tk = Tokenizer(lower=True, filters='', num_words=max_features, oov_token=True)
    tk.fit_on_texts(X_train)
    train_tokenized = tk.texts_to_sequences(X_train)
    valid_tokenized = tk.texts_to_sequences(X_dev)
    test_tokenized = tk.texts_to_sequences(X_test)
    X_train = pad_sequences(train_tokenized, maxlen=max_len)
    X_valid = pad_sequences(valid_tokenized, maxlen=max_len)
    X_test = pad_sequences(test_tokenized, maxlen=max_len)
    embedding_matrix = create_embedding_matrix(args.vec_scheme, tk, max_features)

    model = cnn_attention_network(X_train, y_train, X_valid, y_dev, max_len, max_features, embed_size, embedding_matrix,
                      lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4)

    logging.info("cnn_attention_network")
    logging.info("dataset name: %s", args.train_data)
    f1, score_time = network_pred_f1(model, X_test, y_test)
    logging.info("F1 Score: %s", f1)
    logging.info("Score_time : %s", score_time)

    model = cnn_lstm_network(X_train, y_train, X_valid, y_dev, max_len, max_features, embed_size, embedding_matrix,
                      lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4)

    logging.info("cnn_lstm_network")
    logging.info("dataset name: %s", args.train_data)
    f1, score_time = network_pred_f1(model, X_test, y_test)
    logging.info("F1 Score: %s", f1)
    logging.info("Score_time : %s", score_time)

    model = lstm_network(X_train, y_train, X_valid, y_dev, max_len, max_features, embed_size, embedding_matrix,
                      lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4)

    logging.info("lstm_network")
    logging.info("dataset name: %s", args.train_data)
    f1, score_time = network_pred_f1(model, X_test, y_test)
    logging.info("F1 Score: %s", f1)
    logging.info("Score_time : %s", score_time)

    model = cnn_deep_lstm(X_train, y_train, X_valid, y_dev, max_len, max_features, embed_size, embedding_matrix,
                      lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4)

    logging.info("cnn_deep_lstm")
    logging.info("dataset name: %s", args.train_data)
    f1, score_time = network_pred_f1(model, X_test, y_test)
    logging.info("F1 Score: %s", f1)
    logging.info("Score_time : %s", score_time)


    model = bilstm_network(X_train, y_train, X_valid, y_dev, max_len, max_features, embed_size, embedding_matrix,
                      lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4)

    logging.info("bilstm_network")
    logging.info("dataset name: %s", args.train_data)
    f1, score_time = network_pred_f1(model, X_test, y_test)
    logging.info("F1 Score: %s", f1)
    logging.info("Score_time : %s", score_time)

    model = bilstm_attention_network(X_train, y_train, X_valid, y_dev, max_len, max_features, embed_size, embedding_matrix,
                      lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4)

    logging.info("bilstm_attention_network")
    logging.info("dataset name: %s", args.train_data)
    f1, score_time = network_pred_f1(model, X_test, y_test)
    logging.info("F1 Score: %s", f1)
    logging.info("Score_time : %s", score_time)

    f1, pred_time = bert_network(X_train, y_train, X_valid, y_dev, X_test, y_test, epochs=10)

    logging.info("F1 Score: %s", f1)
    logging.info("Score_time : %s", pred_time)



if __name__ == '__main__':
    main()







