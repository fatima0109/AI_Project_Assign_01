"""
Baseline B: BiLSTM model definition
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPool1D

def build_bilstm(vocab_size, max_len, num_classes, embedding_dim=128):
    """Build BiLSTM model."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    return model