import zipfile
from pathlib import Path
import pandas as pd

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

ROOT = Path(__file__).parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

ZIP_PATH = ROOT / "QEvasion.zip"  # if you unzipped, you can point directly to files instead
# If you already unzipped, set USE_ZIP=False and set TRAIN_PARQUET / TEST_PARQUET paths accordingly
USE_ZIP = False

if USE_ZIP:
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open('QEvasion/data/train-00000-of-00001.parquet') as f:
            train = pd.read_parquet(f)
        with z.open('QEvasion/data/test-00000-of-00001.parquet') as f:
            test = pd.read_parquet(f)
else:
    train = pd.read_parquet(ROOT / "QEvasion" / "data" / "train-00000-of-00001.parquet")
    test = pd.read_parquet(ROOT / "QEvasion" / "data" / "test-00000-of-00001.parquet")

print("Train rows:", len(train))
print("Test rows:", len(test))
print("Columns:", train.columns.tolist())

#normalizing label column name
if 'label' in train.columns and 'clarity_label' not in train.columns:
    train = train.rename(columns={'label':'clarity_label'})
    test = test.rename(columns={'label':'clarity_label'})

#saving sample CSVs
train.head(50).to_csv(OUT / "train_sample50.csv", index=False)
test.head(50).to_csv(OUT / "test_sample50.csv", index=False)

#missing / inaudible summary
summary = {
    "train_total": len(train),
    "train_question_missing": int(train['question'].isna().sum()),
    "train_answer_missing": int(train['interview_answer'].isna().sum()),
    "train_inaudible": int(train['inaudible'].sum()) if 'inaudible' in train.columns else 0,
    "train_multiple_questions": int(train['multiple_questions'].sum()) if 'multiple_questions' in train.columns else 0
}
pd.DataFrame(list(summary.items()), columns=['metric','value']).to_csv(OUT / "train_missing_summary.csv", index=False)

print("Saved train_sample50.csv and train_missing_summary.csv to plots/")
