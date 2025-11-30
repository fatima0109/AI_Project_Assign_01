# Student B: Bi-LSTM (Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPool1D

def build_model(vocab_size, max_len, embedding_dim=100, num_classes=2):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax' if num_classes>2 else 'sigmoid')
    ])
    return model
