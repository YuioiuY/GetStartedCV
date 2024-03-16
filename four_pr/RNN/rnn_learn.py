import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical

#region Const

maxWordsCount = 1000 # max number of words
inp_words = 3 # max input words

#endregion


""" cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print(cwd)
 """

# read file 
with open('four_pr\\RNN\\text.txt', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

#region data processing
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([texts])

dist = list(tokenizer.word_counts.items())
print(dist[:10])

data = tokenizer.texts_to_sequences([texts])
res = np.array( data[0] )
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words] for i in range(n)]) # for model
Y = to_categorical(res[inp_words:], num_classes=maxWordsCount) # for model
#endregion

model = Sequential()

model.add(Embedding(maxWordsCount, 256))
model.add(SimpleRNN(128, activation='sigmoid',return_sequences=True))
model.add(SimpleRNN(128, activation='sigmoid',return_sequences=True))
model.add(SimpleRNN(64, activation='sigmoid'))
model.add(Dense(maxWordsCount, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=50)
model.save('four_pr\\RNN\\RNN_Model.keras')
