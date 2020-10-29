import pandas as pd

import numpy as np

filepath = 'data/sentence_labelled.txt'

df = pd.read_csv(filepath, names=['sentence', 'tag'], sep=',')
print(df.iloc[0])

from sklearn.model_selection import train_test_split

tags = {
    'header':0,
    'document':1,
    'paragraph':2,
    'topic':3,
    'section':4,
    'subsection':5,
    'li':6,
    'footer':7,
    'page_number':8,
    'figure':9,
    'table':10,
    'table_li':11,
    'commentary':12,
    '?':13,
}
decode_tags = {v: k for k, v in tags.items()}
num_tags = max(tags.values()) + 1

sentences = df['sentence'].values
y = []
iterator = 0
for value in df['tag']:
    labels = np.zeros((num_tags,), np.float32)
    labels[tags[value]] = 1
    y.append(labels)

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1 

from keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


Y_train = np.array(y_train)
Y_test = np.array(y_test)

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

from keras.models import Sequential
from keras import layers

embedding_dim = 50
embedding_matrix = create_embedding_matrix('data/glove.6B/glove.6B.50d.txt', tokenizer.word_index, embedding_dim)

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
#model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
#model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(len(tags), activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train, epochs=20, verbose=False, validation_data=(X_test, Y_test), batch_size=10)

loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_history(history)

model_json = model.to_json()
with open("data/model.json", "w") as json_file:
    json_file.write(model_json)

model.save("data/model.h5")

import pickle

with open('data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)