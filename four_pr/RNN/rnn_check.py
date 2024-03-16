import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory

with open('four_pr\\RNN\\text.txt', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([texts])

model = tf.keras.models.load_model('four_pr\\RNN\\RNN_Model.keras')

def buildPhrase(texts, str_len=50):
    
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_len):
        x = data[i: i + len(texts)]
        inp = np.expand_dims(x, axis=0)

        pred = model.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res += " " + tokenizer.index_word[indx]  # дописываем строку

    return res


print('Так как дата сет собран из случайных и никак несвязанных статей, желательно открыть txt файл и взять три слова оттуда )')
while True:
    input_res = str(input("3 слова: "))
    res = buildPhrase(input_res)
    print(res)