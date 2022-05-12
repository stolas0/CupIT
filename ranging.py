import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

np.random.seed(121)


def text2int(data):

    tokenizer = Tokenizer(num_words=3000)
    tokenizer.fit_on_texts(list(data))

    num_words = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(data)

    return num_words, sequences


def run(train, test):

    # concatenate train and test data
    concat_data = train['text'].append(test['text'])

    # turn texts into sequences of integers and  get vocabulary size
    vocab_size, token_text = text2int(concat_data)

    # convert data to appropriate format
    text = sequence.pad_sequences(token_text, padding='post')
    train_score = LabelEncoder().fit_transform(train.iloc[:, 1].values)

    # separate train and test data
    train_text = text[:len(train['text'])]
    test_text = text[len(train['text']):]

    # separating train data into train and valid
    x_train, x_valid, y_train, y_valid = train_test_split(train_text, train_score, test_size=0.1, random_state=0)

    # define the model
    model = Sequential()

    model.add(Embedding(vocab_size, 16, input_length=len(text[0])))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=3, verbose=1)

    # get predicted results
    predicts = model.predict_proba(test_text)
    test_score = [scr[0] for scr in predicts]

    # write to .csv
    sub = pd.DataFrame({'index': range(0, len(predicts)), 'score': test_score})
    sub.to_csv('ranking_answer.csv', index=False)
