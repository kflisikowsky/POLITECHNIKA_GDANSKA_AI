import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# ---- 1. Dane treningowe ----
texts = [
    "sztuczna inteligencja zmienia świat",
    "sztuczna inteligencja tworzy nowe możliwości",
    "model językowy przewiduje słowa",
    "model językowy uczy się z danych"
]

# ---- 2. Tokenizacja ----
tok = Tokenizer()
tok.fit_on_texts(texts)
sequences = []

for line in texts:
    tokens = tok.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        sequences.append(tokens[:i+1])

# np. [sztuczna, inteligencja] -> target: inteligencja

max_len = max(len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")

sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]

vocab_size = len(tok.word_index) + 1

# ---- 3. Model językowy ----
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_len-1),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(vocab_size, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.summary()

# ---- 4. Trening ----
model.fit(X, y, epochs=30, verbose=0)

# ---- 5. Funkcja generowania słów ----
def predict_next(text, n=3):
    for _ in range(n):
        seq = tok.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_len-1, padding="pre")
        pred = model.predict(seq, verbose=0).argmax()
        word = tok.index_word.get(pred, "")
        text += " " + word
    return text

print(predict_next("model językowy"))
print(predict_next("sztuczna inteligencja"))
