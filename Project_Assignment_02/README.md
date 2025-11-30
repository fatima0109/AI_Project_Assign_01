Assignment 02 - Baseline Pipeline (Colab-ready)
==============================================

Contents:
- code/ : training and evaluation scripts and baseline model definitions.
- assignment2_report.tex : LaTeX report (2-3 pages) with Author Contribution Table.
- requirements.txt : Python packages to install in Colab.
- run_in_colab.sh : helper commands to run in Colab (see below).

How to run in Google Colab (recommended)
1. Upload your `Project Assignment 01.zip` to Colab or mount Google Drive, then unzip it:
   - Example Colab cell:
     ```python
     from google.colab import files
     uploaded = files.upload()  # upload Project Assignment 01.zip
     !unzip -q "Project Assignment 01.zip" -d "/content/Project_Assignment_01"
     ```
2. Install dependencies (Colab cell):
   ```bash
   !pip install -q pyarrow pandas scikit-learn matplotlib joblib tensorflow
   !apt-get -qq install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended


Run training (adjust paths):

!python code/train.py --train_path "/content/Project_Assignment_01/complete code files/QEvasion/data/train-00000-of-00001.parquet" \
                     --test_path  "/content/Project_Assignment_01/complete code files/QEvasion/data/test-00000-of-00001.parquet" \
                     --out_dir "/content/Assignment_02_Submission"


Run evaluation to create PDF plots:

!python code/evaluate.py --out_dir "/content/Assignment_02_Submission" --test_path "/content/Project_Assignment_01/complete code files/QEvasion/data/test-00000-of-00001.parquet"


Compile LaTeX report (optional in Colab):

!cp assignment2_report.tex /content/Assignment_02_Submission/
!pdflatex -interaction=nonstopmode -output-directory="/content/Assignment_02_Submission" /content/Assignment_02_Submission/assignment2_report.tex


Zip and download:

!zip -r Assignment_02_Submission.zip /content/Assignment_02_Submission
from google.colab import files; files.download('Assignment_02_Submission.zip')


Author contribution (example):

Student A: Baseline A (TF-IDF + Logistic Regression), EDA: token-length histogram.

Student B: Baseline B (Bi-LSTM), training scripts for Bi-LSTM, loss curve.

Student C: Baseline C (TF-IDF + LinearSVC), baseline comparison table and plots.
