
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from keras.utils import pad_sequences
import numpy as np


nltk.download('stopwords')

stop = stopwords.words('english')

def pre_process_text_data(text):
    # Normalize and remove special characters
    text = text.lower()
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    
    # Remove stop words
    words = text.split()
    words = [w for w in words if w not in stop]
    words = ' '.join(words)
    
    return words

def process_data(df):
    # Preprocess text data
    df['text'] = df['text'].apply(pre_process_text_data)
    
    # Count words
    counts = Counter()
    for i, review in enumerate(df['text']):
        counts.update(review.split())

    words = sorted(counts, key=counts.get, reverse=True)
    
    return df, words

def text_to_int(text, word_to_int):
    return [word_to_int[word] for word in text.split()]

def int_to_text(int_arr, int_to_word):
    return ' '.join([int_to_word[index] for index in int_arr if index != 0])

def map_reviews(df, word_to_int):
    mapped_reviews = []
    for review in df['text']:
        mapped_reviews.append(text_to_int(review, word_to_int))
    
    return mapped_reviews

def get_sequence_length(mapped_reviews):
    length_sent = [len(review) for review in mapped_reviews]
    sequence_length = max(length_sent)
    
    return sequence_length

def pad_and_encode(mapped_reviews, sequence_length):
    X = pad_sequences(maxlen=sequence_length, sequences=mapped_reviews, padding="post", value=0)
    
    return X

def process_sentiment_data(df):
    df, words = process_data(df)
    
    word_to_int = {word: i for i, word in enumerate(words, start=1)}
    int_to_word = {i: word for i, word in enumerate(words, start=1)}
    
    mapped_reviews = map_reviews(df, word_to_int)
    
    sequence_length = get_sequence_length(mapped_reviews)
    
    X = pad_and_encode(mapped_reviews, sequence_length)
    
    y = df['airline_sentiment'].values
    
    return X, y, word_to_int, int_to_word

import numpy as np
from collections import Counter

def pre_process(text: str) -> str:
    text = text.lower()
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text

def get_input_and_labels(text: str, seq_length: int = 10, step: int = 1):
    input_words = []
    label_words = []

    text_arr = text.split()

    for i in range(0, len(text_arr) - seq_length, step):
        x = text_arr[i : (i + seq_length)]
        y = text_arr[i + seq_length]
        input_words.append(x)
        label_words.append(y)

    return input_words, label_words

def process_text_generation_data(text: str, seq_length: int = 10):
    processed_text = pre_process(text)
    input_words, label_words = get_input_and_labels(processed_text, seq_length=seq_length)
    
    counts = Counter()
    counts.update(processed_text.split())
    words = sorted(counts, key=counts.get, reverse=True)
    nb_words = len(processed_text.split())

    word2index = {word: i for i, word in enumerate(words)}
    index2word = {i: word for i, word in enumerate(words)}

    total_words = len(set(words))

    X = np.zeros((len(input_words), seq_length, total_words), dtype=bool)
    y = np.zeros((len(input_words), total_words), dtype=bool)

    for i, input_word in enumerate(input_words):
        for j, word in enumerate(input_word):
            X[i, j, word2index[word]] = 1
        y[i, word2index[label_words[i]]] = 1

    return X, y, words, nb_words, total_words, word2index, index2word, input_words
