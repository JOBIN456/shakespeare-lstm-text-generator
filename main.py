import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# dataset
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

input_sequences = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

input_sequences = np.array(input_sequences)


X = input_sequences[:, :-1]
y = input_sequences[:, -1]   # <-- KEEP AS INTEGER (no one-hot)

model = Sequential([
    Embedding(total_words, 100, input_length=max_seq_len-1),
    LSTM(150),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',   # <-- sparse version
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X, y, epochs=20, batch_size=128)

model.save("shakespeare_model.keras")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully!")