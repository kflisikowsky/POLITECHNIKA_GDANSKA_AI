# TRANSFORMER
# Klasyfikacja sentymentu IMDB z własnoręcznie zbudowanym blokiem Transformera:
#  - Positional Embedding
#  - Multi-Head Attention (Keras)
#  - Feed-Forward Network
#  - Residual + LayerNorm
#  - Zapisywanie i prezentacja wag UWAGI (attention)

# Kod gotowy do uruchomienia w Pythonie/Colabie.


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras


# ------------------------------------------------------------
# PARAMETRY
# ------------------------------------------------------------
VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
EPOCHS = 3
BATCH_SIZE = 128


# ------------------------------------------------------------
# POZYCYJNE EMBEDDINGI
# ------------------------------------------------------------
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(max_len, embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(x)
        return token_embeddings + pos_embeddings

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)  # True = real token


# ------------------------------------------------------------
# BLOK TRANSFORMERA (encoder)
# ------------------------------------------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        # Correctly pass num_heads from init parameters to MultiHeadAttention
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(0.1)
        self.drop2 = layers.Dropout(0.1)

        self.last_attention = None  # do demonstracji wizualnej

    def call(self, inputs, mask=None, training=False):
        attn_out, attn_scores = self.att(
            inputs, inputs,
            attention_mask=mask,
            return_attention_scores=True
        )
        self.last_attention = attn_scores

        attn_out = self.drop1(attn_out, training=training)
        out1 = self.norm1(inputs + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(out1 + ffn_out)


# ------------------------------------------------------------
# MODEL TRANSFORMERA
# ------------------------------------------------------------
def build_model():
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int32")
    x = PositionalEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)(inputs)

    # Create a 2D padding mask (True for real tokens, False for padding)
    padding_mask = keras.ops.not_equal(inputs, 0) # Shape: (batch_size, MAX_LEN)

    # Expand the 2D padding mask to a 3D mask suitable for MultiHeadAttention
    # This creates a mask of shape (batch_size, MAX_LEN, MAX_LEN)
    # True where both query and key tokens are not padding.
    query_mask = keras.ops.expand_dims(padding_mask, axis=-1)  # Shape: (batch_size, MAX_LEN, 1)
    key_mask = keras.ops.expand_dims(padding_mask, axis=-2)   # Shape: (batch_size, 1, MAX_LEN)
    attention_mask = keras.ops.logical_and(query_mask, key_mask) # Shape: (batch_size, MAX_LEN, MAX_LEN)

    tblock = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
    x = tblock(x, mask=attention_mask)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model, tblock


# ------------------------------------------------------------
# ŁADOWANIE DANYCH
# ------------------------------------------------------------
def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)
    return (x_train, y_train), (x_test, y_test)


# ------------------------------------------------------------
# DEKODOWANIE TEKSTU
# ------------------------------------------------------------
def decode_imdb(sequence):
    word_index = imdb.get_word_index()
    index_word = {idx + 3: w for w, idx in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    return " ".join(index_word.get(i, "<UNK>") for i in sequence)


# ------------------------------------------------------------
# DEMO PREDYKCJI + ATTENTION
# ------------------------------------------------------------
def demo_prediction(model, block, text):
    print("\n===== DEMO TEKST =====")
    print(text)

    word_index = imdb.get_word_index()
    seq = []
    for w in text.lower().split():
        idx = word_index.get(w, 2)
        seq.append(idx + 3)

    seq = pad_sequences([seq], maxlen=MAX_LEN)

    pred = model.predict(seq, verbose=0)[0][0]
    label = "POZYTYWNY" if pred >= 0.5 else "NEGATYWNY"
    print("Sentyment:", label, f"(p={pred:.3f})")

    att = block.last_attention
    if att is not None:
        att = tf.reduce_mean(att[0], axis=0).numpy()
        top = np.argsort(att)[-10:][::-1]

        decoded = decode_imdb(seq[0])

        print("\nNajwa\u017Cniejsze tokeny wg attention:")
        words = decoded.split()
        for idx in top:
            if idx < len(words):
                print(f"{words[idx]:15s}  attention={att[idx]:.3f}")


# ------------------------------------------------------------
# G\u0141\u00D3WNY PROGRAM
# ------------------------------------------------------------
def main():
    print("\u0141adowanie IMDB\u2026")
    (x_train, y_train), (x_test, y_test) = load_data()

    print("Budowanie modelu\u2026")
    model, block = build_model()
    model.summary()

    print("\nTrening\u2026")
    model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAccuracy na te\u015Bcie: {acc:.4f}")

    # Demo predykcji
    demo_prediction(model, block,
                    "This movie was absolutely wonderful, I loved every moment.")
    demo_prediction(model, block,
                    "This film was boring, slow and painful to watch.")


if __name__ == "__main__":
    main()
