#! usr/bin/env python
# -*- coding : utf-8 -*-

import tensorflow as tf
import codecs
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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
import time
from utils import load_tab_data
from args import get_args
import numpy as np
import os
import wget
import zipfile
import logging
from glob import glob
logging.basicConfig(filename='../logs/network_models_results.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  datefmt='%H:%M:%S', level=logging.DEBUG)



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






strategy = tf.distribute.MirroredStrategy()


def cnn_attention_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"../models/best_model_cnn_network.hdf5"
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
    if not args.only_test:
        model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    with strategy.scope():
        model2 = Model(inputs=inp, outputs=x)
        model2.load_weights(file_path)
        model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def cnn_lstm_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"../models/best_model_cnn_lstm_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    model = Sequential()
    model.add(Embedding(max_features + 1, embed_size, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Conv1D(200, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(100))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(1, activation='sigmoid'))
    model2 = model
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    if not args.only_test:
        model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def lstm_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"../models/best_model_lstm_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape = (max_len,),name='main_input')
    glove_Embed = (Embedding(max_features + 1, embed_size , weights=[embedding_matrix], trainable=False))(main_input)
    y = LSTM(300)(glove_Embed)
    y = Dense(200, activation='relu')(y)
    y = Dropout(rate=0.15)(y)
    z = Dense(100, activation='relu')(y)
    output_lay = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[main_input], outputs=[output_lay])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model2 = Model(inputs=[main_input], outputs=[output_lay])
    if not args.only_test:
        model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def cnn_deep_lstm(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"../models/best_model_cnn_deep_lstm.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    glove_Embed = (Embedding(max_features + 1, embed_size, input_length=max_len, weights=[embedding_matrix], trainable=False))(main_input)

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
    if not args.only_test:
        model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def bilstm_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"../models/best_model_bilstm_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    x = (Embedding(max_features + 1, embed_size, input_length=max_len, weights=[embedding_matrix], trainable=False))(main_input)
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
    if not args.only_test:
        model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    model2.load_weights(file_path)
    model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2


def bilstm_attention_network(X_train, y_train, X_valid, y_valid, max_len, max_features, embed_size, embedding_matrix, lr=0.0, lr_d=0.0, spatial_dr=0.0, dense_units=128, conv_size=128, dr=0.2, patience=3):
    file_path = f"../models/best_model_bilstm_attention_network.hdf5"
    check_point = ModelCheckpoint(file_path, monitor="val_accuracy", verbose=1,save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    with strategy.scope():
        main_input = Input(shape=(max_len,), name='main_input')
        x = (Embedding(max_features + 1, embed_size, input_length=max_len, weights=[embedding_matrix], trainable=False))(main_input)
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
    if not args.only_test:
        model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), verbose=1, callbacks=[early_stop, check_point])
    with strategy.scope():
        model2.load_weights(file_path)
        model2.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    return model2









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

def test_files(model, tk, model_name):
    for test_file in glob(os.path.join(args.test_path, '*test_data*obfuscated.txt')):
        X_test_o, y_test = load_tab_data(filename=test_file, preprocessed=True, test_file=True)
        test_tokenized = tk.texts_to_sequences(X_test_o)
        X_test = pad_sequences(test_tokenized, maxlen=args.max_len)
        loaded_model = model
        a = time.time()
        y_preds = loaded_model.predict(X_test)
        score_time = time.time() - a
        result_file = os.path.join(args.evaluate_label_path, os.path.basename(test_file).replace('.txt', '') + '_' +
                                   os.path.basename(args.train_data) + '_' + model_name + '_predictions.txt')

        with codecs.open(result_file, 'w', 'utf-8') as result_obj:
            for i, val in enumerate(X_test):
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
    max_features = args.max_features
    max_len = args.max_len
    embed_size = args.embed_size
    X_train_o, y_train = load_tab_data(filename=args.train_data, preprocessed=True)


    X_dev_o, y_dev = load_tab_data(filename=args.dev_data, preprocessed=True)

    tk = Tokenizer(lower=True, filters='', num_words=max_features, oov_token=True)
    tk.fit_on_texts(X_train_o)
    train_tokenized = tk.texts_to_sequences(X_train_o)
    valid_tokenized = tk.texts_to_sequences(X_dev_o)
    X_train = pad_sequences(train_tokenized, maxlen=max_len)
    X_valid = pad_sequences(valid_tokenized, maxlen=max_len)

    embedding_matrix = create_embedding_matrix(args.vec_scheme, tk, max_features)




    # BILSTM ATTENTION

    a = time.time()
    model = bilstm_attention_network(X_train, y_train, X_valid, y_dev, max_len, max_features, embed_size, embedding_matrix, lr=1e-3, lr_d=0, spatial_dr=0.1, dense_units=128, conv_size=128, dr=0.1, patience=4)
    fit_time = time.time() - a
    logging.info("=================================START====================================\n")
    logging.info("==============================TRAINING====================================")
    logging.info("bilstm_attention_network")
    logging.info("dataset name: %s", args.train_data)
    logging.info("fit_time: %s", fit_time)
    logging.info("\n")
    logging.info("==========================TESTING==========================================")
    test_files(model, tk, 'bilstm_attention_network')




if __name__ == '__main__':
    main()







