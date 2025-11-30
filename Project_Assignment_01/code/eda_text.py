# eda_text.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

def load_data():
    
    parquet_path = Path("QEvasion/data/train-00000-of-00001.parquet")
    train = pd.read_parquet(parquet_path)

    if 'label' in train.columns and 'clarity_label' not in train.columns:
        train = train.rename(columns={'label':'clarity_label'})
    return train

train = load_data()

# token counts
train['question_word_count'] = train['question'].fillna("").astype(str).apply(lambda s: len(s.split()))
train['answer_word_count'] = train['interview_answer'].fillna("").astype(str).apply(lambda s: len(s.split()))

# save histograms
plt.figure(figsize=(6,4)); train['answer_word_count'].hist(bins=40); plt.title("Train: Answer word count")
plt.xlabel("Words"); plt.tight_layout(); plt.savefig(OUT / "train_answer_wordcount_hist.pdf"); plt.close()

plt.figure(figsize=(6,4)); train['question_word_count'].hist(bins=30); plt.title("Train: Question word count")
plt.xlabel("Words"); plt.tight_layout(); plt.savefig(OUT / "train_question_wordcount_hist.pdf"); plt.close()

# ngrams
def top_ngrams(texts, n=1, k=30):
    c = Counter()
    for t in texts.fillna("").astype(str):
        toks = re.findall(r"\w+", t.lower())
        if n==1:
            c.update(toks)
        else:
            c.update([" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)])
    return c.most_common(k)

q_uni = top_ngrams(train['question'],1,30)
a_uni = top_ngrams(train['interview_answer'],1,30)
q_bi = top_ngrams(train['question'],2,30)
a_bi = top_ngrams(train['interview_answer'],2,30)

pd.DataFrame(q_uni, columns=['token','count']).to_csv(OUT / "train_question_unigrams.csv", index=False)
pd.DataFrame(a_uni, columns=['token','count']).to_csv(OUT / "train_answer_unigrams.csv", index=False)
pd.DataFrame(q_bi, columns=['token','count']).to_csv(OUT / "train_question_bigrams.csv", index=False)
pd.DataFrame(a_bi, columns=['token','count']).to_csv(OUT / "train_answer_bigrams.csv", index=False)

print("Saved histograms and n-gram CSVs to plots/")
