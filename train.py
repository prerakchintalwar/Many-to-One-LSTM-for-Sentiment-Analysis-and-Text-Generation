import os
from collections import Counter
import tensorflow as tf
from keras.layers import Dense, Activation, SimpleRNN, LSTM, GRU
from keras.models import Sequential
from keras.utils import to_categorical, pad_sequences
from keras.layers import Embedding
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

pd.set_option('display.max_colwidth', None)
tf.keras.backend.set_image_data_format("channels_last")
nltk.download('stopwords')

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def train_sentiment_lstm(X_train, y_train, X_test, y_test,word_to_int, embedding_dim=32, lstm_units=40, epochs=50, batch_size=32):
    # Create the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=len(word_to_int)+1, output_dim=embedding_dim, input_length=X_train.shape[1]))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(LSTM(units=lstm_units, return_sequences=False))
    model.add(Dense(units=2, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Print model summary
    print(model.summary())
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    
    # Print training history
    print('Training Loss:', history.history['loss'])
    print('Validation Loss:', history.history['val_loss'])
    print('Training Accuracy:', history.history['accuracy'])
    print('Validation Accuracy:', history.history['val_accuracy'])
    
    # Save the model
    model.save('output/sentiment_model.h5')
    return model

SEQLEN = 10
STEP = 1

import numpy as np
import tensorflow as tf

def create_text_generation_model(hidden_size, seq_length, total_words):
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=False, input_shape=(seq_length, total_words)))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


BATCH_SIZE = 32
NUM_ITERATIONS = 100
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100


def check_model_output(model, preds: int, input_words, seq_length, total_words, word2index, index2word):
    test_idx = np.random.randint(int(len(input_words) * 0.1)) * (-1)
    test_words = input_words[test_idx]

    for curr_pred in range(preds):
        curr_embedding = np.zeros((1, seq_length, total_words))

        for i, ch in enumerate(test_words):
            curr_embedding[0, i, word2index[ch]] = 1

        pred = model.predict(curr_embedding, verbose=0)[0]
        word_pred = index2word[np.argmax(pred)]

        print("=" * 50)
        print(f"Prediction {curr_pred + 1} of {preds}")
        print(f'Generating from seed: {" ".join(test_words)}\nNext Word: {word_pred}')
        print("=" * 50)

        test_words = test_words[1:] + [word_pred]

def train_text_generation_model(X, y, input_words, seq_length, total_words, model, word2index, index2word):
    
    for iteration in range(NUM_ITERATIONS):
        model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION, validation_split=0.1)
        if iteration % 10 == 0:
            check_model_output(model, 5, input_words, seq_length, total_words, word2index, index2word)
    # Save the trained model
    model.save('output/text_gen_model.h5')
    return model



def predict_next_word(model, input_text: str, seq_length, total_words, word2index, index2word, temperature=None):
    curr_embedding = np.zeros((1, seq_length, total_words))

    for i, ch in enumerate(input_text):
        curr_embedding[0, i, word2index[ch]] = 1

    pred = model.predict(curr_embedding, verbose=0)[0]

    if temperature == None:
        word_pred = index2word[np.argmax(pred)]
    else:
        next_word_token = tf.random.categorical(tf.expand_dims(pred / temperature, 0), num_samples=1)[-1, 0].numpy()
        word_pred = index2word[next_word_token]

    return pred, word_pred

def generate_paragraph(model, seed, words: int, temperature: int, total_words, word2index, index2word):
    full_text = seed.copy()
    for _ in range(words):
        logits, word_pred = predict_next_word(model, seed, SEQLEN, total_words, word2index, index2word, temperature=temperature)
        seed = (seed + [word_pred])[-10:]
        full_text = full_text + [word_pred]
    return full_text

