**Sentiment Analysis in Twitter**
SemEval-2017 Task 4 — Message-Level Sentiment Classification
MCA (Data Science & Informatics), NIT Patna
Overview:-

This repository contains the implementation of Sentiment Analysis on Twitter based on the SemEval-2017 Task 4 benchmark.
The task focuses on message-level sentiment classification — categorizing tweets as Positive, Negative, or Neutral.

This implementation uses a modern Transformer-based approach (DistilBERT) fine-tuned on the TweetEval dataset, a cleaned version of the official SemEval Twitter dataset.
All experiments were performed using Google Colab and Hugging Face Transformers.

Key Features:-

Preprocessing of raw tweets (removing URLs, mentions, hashtags)
Stratified sampling for balanced training and validation sets
Fine-tuning DistilBERT for 3-class sentiment classification
Evaluation using Accuracy, Weighted F1, and Confusion Matrix
Fully Colab-compatible (no API keys, no wandb setup needed)

Dataset:-
Primary Source

SemEval-2017 Task 4: Sentiment Analysis in Twitter
Subtask A: Message-level sentiment classification
Official Task Page

Fallback Dataset:-
TweetEval Sentiment Dataset (Hugging Face)
Used automatically if the original SemEval dataset isn’t found in Drive.
Each entry in the dataset contains:
text — the tweet content
label — 0 = negative, 1 = neutral, 2 = positive
🛠️Setup and Requirements
1️⃣ Install Dependencies
pip install -q transformers datasets accelerate evaluate scikit-learn
2️⃣ (Optional) Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')
3️⃣ Load Dataset Automatically
If SemEval TSV files are found in Drive, they’ll be used; otherwise, TweetEval is loaded automatically.

Model Architecture:-
Component	Description
Base Model	distilbert-base-uncased
Task	3-class text classification
Training	Fine-tuning with Hugging Face Trainer API
Loss Function	Cross-Entropy
Optimizer	AdamW
Epochs	1–2 (configurable)
Batch Size	8
Max Sequence Length	64 tokens
Training & Evaluation:-
# Start training
trainer.train()
# Evaluate on dev set
trainer.evaluate()
# Evaluate on test set
preds = trainer.predict(hf_test)

Metrics:
Accuracy
Weighted F1 Score
Macro F1 Score

Confusion Matrix Example (Dev Set):

[[420  30  50]
 [ 40 310  35]
 [ 20  45 470]]

Results Summary:-
Metric	Dev Set
Accuracy	0.83
Weighted F1	0.82
Macro F1	0.79

The model performs best on positive and negative sentiments, with minor overlap in neutral predictions.
🧠 Technologies Used:-
Python 3.10
Transformers (Hugging Face)
Datasets
Scikit-learn
Matplotlib
Google Colab
Citation:-
If you refer to the original benchmark task or dataset, please cite:
@InProceedings{rosenthal-etal-2017-semeval,
  title     = {SemEval-2017 Task 4: Sentiment Analysis in Twitter},
  author    = {Rosenthal, Sara and Farra, Noura and Nakov, Preslav},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {502--518}
}

Authors:-
Rajendra Kumar Ojha (Roll No: 2446040)
Abhishek Kumar (Roll No:2446047)
Mohammed Irfan (Roll No:-2446046)
Department of Computer Applications
National Institute of Technology, Patna
