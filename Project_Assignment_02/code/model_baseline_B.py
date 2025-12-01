import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np


# LOAD DATASET
DATASET_PATH = r"C:\Users\Shining star\Desktop\AI\Project Assignments\Baseline Pipeline Implementation\QEvasion\train\train"
dataset = load_from_disk(DATASET_PATH)

df = dataset.to_pandas()

# CREATE TEXT COLUMN
df["text"] = (
    df["question"].fillna("") + " " +
    df["interview_question"].fillna("") + " " +
    df["interview_answer"].fillna("")
)

# LABELS
df["label"] = df["label"].astype("category").cat.codes
num_classes = df["label"].nunique()

texts = df["text"].tolist()
labels = df["label"].tolist()

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# TOKENIZER
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

y_train = np.array(y_train)
y_test = np.array(y_test)

# BI-LSTM MODEL FOR MULTICLASS
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32
)

print(model.evaluate(X_test_seq, y_test))
