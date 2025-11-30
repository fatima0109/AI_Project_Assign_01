"""
Evaluate saved models and create PDF plots (confusion matrices, comparison bar chart, loss curves if available).
Run after train.py. Example:
python code/evaluate.py --out_dir "/content/Assignment_02_Submission" --test_path "/content/Project_Assignment_01/complete code files/QEvasion/data/test-00000-of-00001.parquet"
"""
import argparse, os, joblib, pickle
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

def save_cm(y_true, y_pred, fname, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(5,4))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(l) for l in sorted(np.unique(y_true))])
    disp.plot(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, format='pdf')
    plt.close(fig)

def main(args):
    out_dir = args.out_dir
    df_test = pd.read_parquet(args.test_path)
    # auto detect cols
    if 'text' not in df_test.columns:
        # try to find textual column
        for c in df_test.columns:
            if df_test[c].dtype == object:
                text_col = c; break
        # try label col
        label_col = 'label' if 'label' in df_test.columns else min(df_test.columns.tolist(), key=lambda c: df_test[c].nunique())
        df_test = df_test[[text_col,label_col]].dropna().rename(columns={text_col:'text', label_col:'label'})
    y_true = df_test['label'].values
    # load models
    pipeA = joblib.load(os.path.join(out_dir,'models','baseline_A.joblib'))
    pipeC = joblib.load(os.path.join(out_dir,'models','baseline_C.joblib'))
    with open(os.path.join(out_dir,'models','tokenizer.pkl'),'rb') as f:
        tokenizer = pickle.load(f)
    # Baseline A/C predictions on test
    predA = pipeA.predict(df_test['text'].values)
    predC = pipeC.predict(df_test['text'].values)
    # Baseline B predictions
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    modelB = load_model(os.path.join(out_dir,'models','baseline_B.keras'))
    max_len = 200
    seq = tokenizer.texts_to_sequences(df_test['text'].values)
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prob = modelB.predict(pad)
    if prob.shape[1] > 1:
        predB = np.argmax(prob, axis=1)
    else:
        predB = (prob>0.5).astype(int).reshape(-1)
    # save confusion matrices
    os.makedirs(os.path.join(out_dir,'plots'), exist_ok=True)
    save_cm(y_true, predA, os.path.join(out_dir,'plots','confusion_A_test.pdf'),'Baseline A (test)')
    save_cm(y_true, predB, os.path.join(out_dir,'plots','confusion_B_test.pdf'),'Baseline B (test)')
    save_cm(y_true, predC, os.path.join(out_dir,'plots','confusion_C_test.pdf'),'Baseline C (test)')
    # metrics summary plot
    accs = [accuracy_score(y_true,predA), accuracy_score(y_true,predB), accuracy_score(y_true,predC)]
    f1s = [f1_score(y_true,predA,average='macro'), f1_score(y_true,predB,average='macro'), f1_score(y_true,predC,average='macro')]
    labels = ['A_TFIDF_LR','B_BiLSTM','C_TFIDF_SVC']
    dfm = pd.DataFrame({'baseline':labels,'accuracy':accs,'f1_macro':f1s})
    dfm.to_csv(os.path.join(out_dir,'results','test_metrics.csv'), index=False)
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(labels))
    ax.bar(x-0.2, dfm['accuracy'], width=0.4, label='Accuracy')
    ax.bar(x+0.2, dfm['f1_macro'], width=0.4, label='F1-macro')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1)
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir,'plots','baseline_comparison_test.pdf'), format='pdf')
    print('Evaluation PDFs saved in', os.path.join(out_dir,'plots'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--test_path', required=True)
    args = p.parse_args()
    main(args)
