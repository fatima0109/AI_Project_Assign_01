"""
Train the three baselines. Run in Google Colab or local with installed deps.

Usage example in Colab (after uploading the ZIP and extracting):
!python code/train.py --train_path "/content/Project_Assignment_01/complete code files/QEvasion/data/train-00000-of-00001.parquet" \
                      --test_path "/content/Project_Assignment_01/complete code files/QEvasion/data/test-00000-of-00001.parquet" \
                      --out_dir "/content/Assignment_02_Submission"
"""
import argparse, os, joblib, pickle
from code.data_loader import load_parquet_dataset
from code.preprocess import split_train_val
from code.model_baseline_A import build_pipeline as build_A
from code.model_baseline_B import build_model as build_B
from code.model_baseline_C import build_pipeline as build_C
import numpy as np
import pandas as pd

def train_tf_baseline(build_fn, X_train, y_train, X_val, y_val, out_path):
    pipe = build_fn()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    joblib.dump(pipe, out_path)
    return y_pred, pipe

def train_bilstm(X_train, y_train, X_val, y_val, out_path, tokenizer_path):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    max_words = 20000; max_len = 200
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train); X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    vocab_size = min(max_words, len(tokenizer.word_index)+1)
    num_classes = len(np.unique(y_train))
    model = build_B(vocab_size, max_len, embedding_dim=100, num_classes=num_classes)
    loss = 'sparse_categorical_crossentropy' if num_classes>2 else 'binary_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_val_pad, y_val))
    model.save(out_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    # predict on val
    y_prob = model.predict(X_val_pad)
    if y_prob.shape[1] > 1:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob>0.5).astype(int).reshape(-1)
    return y_pred, model

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'models'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'results'), exist_ok=True)
    # load
    df_train, df_test = load_parquet_dataset(args.train_path, args.test_path)
    X_train, X_val, y_train, y_val = split_train_val(df_train, val_size=0.15)
    # Baseline A
    y_pred_a, pipe_a = train_tf_baseline(build_A, X_train, y_train, X_val, y_val, os.path.join(args.out_dir,'models','baseline_A.joblib'))
    # Baseline C
    y_pred_c, pipe_c = train_tf_baseline(build_C, X_train, y_train, X_val, y_val, os.path.join(args.out_dir,'models','baseline_C.joblib'))
    # Baseline B (Bi-LSTM)
    y_pred_b, model_b = train_bilstm(X_train, y_train, X_val, y_val, os.path.join(args.out_dir,'models','baseline_B.keras'), os.path.join(args.out_dir,'models','tokenizer.pkl'))
    # Save classification reports
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    with open(os.path.join(args.out_dir,'results','report_baseline_A.txt'),'w') as f:
        f.write(classification_report(y_val, y_pred_a, zero_division=0))
    with open(os.path.join(args.out_dir,'results','report_baseline_C.txt'),'w') as f:
        f.write(classification_report(y_val, y_pred_c, zero_division=0))
    with open(os.path.join(args.out_dir,'results','report_baseline_B.txt'),'w') as f:
        f.write(classification_report(y_val, y_pred_b, zero_division=0))
    # Save metrics CSV
    metrics = {
        'baseline':['A_TFIDF_LR','B_BiLSTM','C_TFIDF_SVC'],
        'val_acc':[accuracy_score(y_val,y_pred_a), accuracy_score(y_val,y_pred_b), accuracy_score(y_val,y_pred_c)],
        'val_f1_macro':[f1_score(y_val,y_pred_a,average='macro'), f1_score(y_val,y_pred_b,average='macro'), f1_score(y_val,y_pred_c,average='macro')]
    }
    pd.DataFrame(metrics).to_csv(os.path.join(args.out_dir,'results','baseline_metrics.csv'), index=False)
    print('Training complete. Outputs in', args.out_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', required=True)
    p.add_argument('--test_path', required=True)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    main(args)
