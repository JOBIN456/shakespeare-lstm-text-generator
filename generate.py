import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model("shakespeare_model.keras")
print("Model loaded successfully!")

# ---------------------------
# LOAD TOKENIZER
# ---------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer loaded successfully!")

# ---------------------------
# IMPORTANT: SET MAX SEQ LENGTH
# (Must be same as training)
# ---------------------------
# If you know the value from training, put it directly.
# Example: max_seq_len = 15
# If not sure, rebuild quickly from dataset.

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_seq_len = max(len(seq) for seq in input_sequences)

# ---------------------------
# TEXT GENERATION FUNCTION
# ---------------------------
def generate_text(seed_text, next_words=50):
    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len-1,
            padding='pre'
        )

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        if output_word == "":
            break

        seed_text += " " + output_word

    return seed_text


# ---------------------------
# INTERACTIVE MODE
# ---------------------------
print("\nType a starting word (example: romeo, king, love)")
user_input = input("Enter seed text: ")

generated = generate_text(user_input, next_words=40)

print("\nGenerated Text:\n")
print(generated)