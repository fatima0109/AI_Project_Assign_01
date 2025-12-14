Project Overview
This repository contains our Assignment 03 implementation for the CLARITY ("I Never Said That") task, focusing on enhanced modeling for political response clarity and evasion detection. Building on Assignment 02, this phase emphasizes deeper exploratory insights, improved feature engineering, class imbalance handling, and systematic model comparison.

Dataset
We use the official CLARITY / “I Never Said That” dataset for political interview analysis.

Files used:
train.parquet – full training set (3,448 samples)
test.parquet – held-out evaluation set (308 samples)
train_sample50.csv – small subset for rapid debugging

Key fields:
question
interview_answer
clarity_label (primary target)
Metadata fields such as inaudible, multiple_questions, and annotator information

Exploratory Data Analysis (Summary)
EDA from Assignment 02 is extended and reused to guide model improvements:
Class imbalance: Clear responses dominate (~62%), with evasive responses forming a small minority (~10%).
Length patterns: Answers show high variance; evasive answers are often longer and include discourse markers.

Data quality issues:
Multi-annotator disagreement (~18%)
Inaudible or clipped answers (~8%)
These observations directly motivate the use of class weighting, n-grams, and improved preprocessing in Assignment 03.

Assignment 03: Enhanced Modeling
What Changed from Assignment 02
Compared to the baseline pipelines, Assignment 03 introduces:
More robust text preprocessing tailored to political discourse
Optimized TF-IDF feature extraction with uni- and bi-grams
Explicit handling of class imbalance
Hyperparameter tuning using cross-validation
Broader model comparison and ensemble evaluation

Models Implemented
Model	Description
Baseline A	TF-IDF + Logistic Regression (class-weighted)
Baseline C	TF-IDF + LinearSVC (balanced SVM)
Proposed Model	Enhanced TF-IDF + Logistic Regression with optimized parameters
Random Forest	TF-IDF features with ensemble trees
Ensemble	Majority voting across multiple classifiers
Technical Setup

Language: Python 3.9
Libraries: scikit-learn, pandas, matplotlib, pyarrow
Training: 5-fold cross-validation
Evaluation Metrics: Accuracy, Precision, Recall, F1-score
Hardware: CPU-based training (Intel i7 class machine)

Results (Test Set)
Model	Accuracy	F1
Baseline A	46.2%	45.5%
Baseline C	46.8%	46.1%
Proposed Model	47.1%	46.4%
Random Forest	44.3%	43.4%
Ensemble	47.5%	46.8%

Key observations:
Incremental gains from better preprocessing and feature tuning
Ensemble voting provides the most stable performance
Minority classes remain challenging due to imbalance
Error Analysis Highlights
Majority classes are predicted reliably
Minority (evasive / unclear) classes show high confusion
Very short or generic answers are frequently misclassified
Discourse markers (e.g., "well", "let me") are strong indicators of evasion

Limitations
Strong label imbalance
No access to audio/video cues
Limited context beyond the immediate Q&A pair
Inconsistent annotations across annotators

Future Work
Planned improvements beyond Assignment 03 include:
Data augmentation for minority classes
Transformer-based encoders (BERT, DeBERTa)
Multimodal fusion (text + audio + visual cues)
Context enrichment using political knowledge graphs

Author Contributions
Fatima Malik: Team coordination, proposed model development, experiments, report writing
Shuja Naveed: Evaluation pipeline, visualizations, analysis support
Sumera Bibi: Ensemble modeling, comparative analysis
