import os
from ml_pipeline import train, process, utils
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from projectpro import save_point, checkpoint

# Define your main engine code here

def main():
    # Code to call the relevant functions from the ml_pipeline modules based on input
    # Example:
    input_type = 2  # Change this based on your input
    if input_type == 1:
        df = pd.read_csv('data/airline_sentiment.csv')
        X, y , word_to_int, int_to_word = process.process_sentiment_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = train.train_sentiment_lstm(X_train, y_train, X_test, y_test, word_to_int)
        positive_sentences, negative_sentences = utils.get_predicted_sentiments(model, X_test, int_to_word)
        checkpoint('5b420a')
        print("Positive Sentences:")
        for sentence in positive_sentences[:5]:
            print(sentence)

        print("\nNegative Sentences:")
        for sentence in negative_sentences[:5]:
            print(sentence)

        
        
    elif input_type == 2:
        # Additional code for text generation
        text = utils.load_data()
        X, y, words, nb_words, total_words, word2index, index2word, input_words = process.process_text_generation_data(text)
        print(f'Input of X: {X.shape}\nInput of y: {y.shape}')
        SEQLEN = 10
        HIDDEN_SIZE = 128
       
        model = model = train.create_text_generation_model(HIDDEN_SIZE, SEQLEN, total_words)
        save_point('5b420a')
        model = train.train_text_generation_model(X, y, input_words, SEQLEN, total_words, model, word2index, index2word)
        test_words = input_words[-28701]

        for _ in range(2):
            print(' '.join(train.generate_paragraph(model, test_words, 12, 10)))

if __name__ == "__main__":
    main()
