# SemEval-2017 Task 4 - Sentiment Analysis on Twitter
# Complete implementation with direct dataset loading

# Install required packages
!pip install -q transformers datasets accelerate evaluate scikit-learn

# Mount Google Drive to access dataset
from google.colab import drive
drive.mount('/content/drive')

import os
import re
import json
import random
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Disable wandb to avoid API key issues
os.environ["WANDB_DISABLED"] = "true"

# Configuration parameters
MODEL_NAME = "distilbert-base-uncased"  # Using DistilBERT for faster training
DATASET_PATH = "/content/drive/MyDrive/Mechine Learning WorkSpace/Projects/datastories-semeval2017-task4-master/dataset"

# Training parameters
REDUCED_TRAIN = 3000   # Number of training samples
REDUCED_DEV = 800      # Number of validation samples  
REDUCED_TEST = 800     # Number of test samples
EPOCHS = 1             # Training epochs
BATCH_SIZE = 8         # Batch size
MAX_LEN = 64           # Maximum sequence length
LEARNING_RATE = 2e-5   # Learning rate
SEED = 42              # Random seed for reproducibility

# Set random seeds for reproducible results
random.seed(SEED)
np.random.seed(SEED)

print("Starting SemEval-2017 Sentiment Analysis Project")
print(f"Using model: {MODEL_NAME}")
print(f"Dataset path: {DATASET_PATH}")

# Function to load dataset files using pandas
def load_semeval_dataset(base_path):
    """
    Load SemEval dataset files directly using pandas
    Returns: train_df, dev_df, test_df as pandas DataFrames
    """
    print("Loading dataset files from:", base_path)
    
    # Define possible file paths
    train_file = os.path.join(base_path, "train.tsv")
    dev_file = os.path.join(base_path, "dev.tsv") 
    test_file = os.path.join(base_path, "test.tsv")
    
    # Alternative file names for SemEval dataset
    alt_train = os.path.join(base_path, "SemEval2017-task4-dev.subtask-A.english.INPUT.txt")
    alt_dev = os.path.join(base_path, "SemEval2017-task4-test.subtask-A.english.INPUT.txt")
    
    train_df, dev_df, test_df = None, None, None
    
    # Load training data
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file, sep='\t', header=0)
        print("Loaded train data from:", train_file)
    elif os.path.exists(alt_train):
        train_df = pd.read_csv(alt_train, sep='\t', header=0)
        print("Loaded train data from:", alt_train)
    else:
        print("Train file not found in dataset directory")
        return None, None, None
    
    # Load development/validation data
    if os.path.exists(dev_file):
        dev_df = pd.read_csv(dev_file, sep='\t', header=0)
        print("Loaded dev data from:", dev_file)
    elif os.path.exists(alt_dev):
        dev_df = pd.read_csv(alt_dev, sep='\t', header=0)
        print("Loaded dev data from:", alt_dev)
    else:
        print("Dev file not found in dataset directory")
        return train_df, None, None
    
    # Load test data
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file, sep='\t', header=0)
        print("Loaded test data from:", test_file)
    else:
        print("Test file not found, continuing without test set")
    
    return train_df, dev_df, test_df

# Function to prepare dataframe by identifying text and label columns
def prepare_dataframe(df):
    """
    Identify and extract text and label columns from dataframe
    Returns: DataFrame with 'text' and 'label' columns
    """
    if df is None:
        return None
        
    # Convert column names to lowercase for easier processing
    df.columns = [col.lower() for col in df.columns]
    
    # Identify text column (look for common names)
    text_col = None
    for col in df.columns:
        if 'text' in col or 'tweet' in col or 'sentence' in col:
            text_col = col
            break
    
    # If not found, use the second column as text (common in TSV files)
    if not text_col and df.shape[1] >= 2:
        text_col = df.columns[1]
    
    # Identify label column (look for common names)
    label_col = None
    for col in df.columns:
        if 'label' in col or 'sentiment' in col or 'class' in col:
            label_col = col
            break
    
    # If not found, use the third column as label or first as fallback
    if not label_col:
        if df.shape[1] >= 3:
            label_col = df.columns[2]
        else:
            label_col = df.columns[0]
    
    # Create new dataframe with standardized column names
    if text_col and label_col:
        result_df = df[[text_col, label_col]].copy()
        result_df.columns = ['text', 'label']
        print(f"Using columns: '{text_col}' as text, '{label_col}' as label")
        return result_df
    else:
        print("Error: Could not identify text and label columns")
        return None

# Load the dataset
train_df, dev_df, test_df = load_semeval_dataset(DATASET_PATH)

# Prepare dataframes with proper column names
if train_df is not None:
    train_df = prepare_dataframe(train_df)
if dev_df is not None:
    dev_df = prepare_dataframe(dev_df)
if test_df is not None:
    test_df = prepare_dataframe(test_df)

# If dataset loading failed, use fallback dataset from Hugging Face
if train_df is None or dev_df is None:
    print("Using fallback dataset: tweet_eval from Hugging Face")
    dataset = load_dataset("tweet_eval", "sentiment")
    train_df = pd.DataFrame({'text': dataset['train']['text'], 'label': dataset['train']['label']})
    dev_df = pd.DataFrame({'text': dataset['validation']['text'], 'label': dataset['validation']['label']})
    test_df = pd.DataFrame({'text': dataset['test']['text'], 'label': dataset['test']['label']}) if 'test' in dataset else None

print(f"Dataset sizes - Training: {len(train_df)}, Validation: {len(dev_df)}, Test: {len(test_df) if test_df is not None else 'Not available'}")

# Text cleaning function
def clean_text(text):
    """
    Clean tweet text by removing URLs, mentions, hashtags and extra spaces
    """
    text = str(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove user mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtag symbols (keep the text)
    text = re.sub(r"#", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply text cleaning to all datasets
print("Cleaning text data...")
train_df['text'] = train_df['text'].apply(clean_text)
dev_df['text'] = dev_df['text'].apply(clean_text)
if test_df is not None:
    test_df['text'] = test_df['text'].apply(clean_text)

# Function for stratified sampling to maintain class distribution
def stratified_sample(df, n_samples, label_col='label'):
    """
    Create stratified sample maintaining original class distribution
    """
    if n_samples >= len(df):
        return df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Calculate samples per class based on original distribution
    class_counts = df[label_col].value_counts()
    samples_per_class = {}
    
    for class_label, count in class_counts.items():
        proportion = count / len(df)
        samples_per_class[class_label] = max(1, int(proportion * n_samples))
    
    # Adjust total samples to match exactly n_samples
    total_allocated = sum(samples_per_class.values())
    
    if total_allocated < n_samples:
        # Add remaining samples to the largest class
        difference = n_samples - total_allocated
        largest_class = max(samples_per_class.items(), key=lambda x: x[1])[0]
        samples_per_class[largest_class] += difference
    elif total_allocated > n_samples:
        # Remove excess samples from the largest class
        difference = total_allocated - n_samples
        largest_class = max(samples_per_class.items(), key=lambda x: x[1])[0]
        samples_per_class[largest_class] = max(1, samples_per_class[largest_class] - difference)
    
    # Sample from each class
    sampled_dfs = []
    for class_label, n in samples_per_class.items():
        class_df = df[df[label_col] == class_label]
        if len(class_df) > 0:
            n_actual = min(n, len(class_df))
            sampled_class = class_df.sample(n=n_actual, random_state=SEED)
            sampled_dfs.append(sampled_class)
    
    # Combine and shuffle
    if sampled_dfs:
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        return result_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        return df.sample(n=min(n_samples, len(df)), random_state=SEED).reset_index(drop=True)

# Create smaller datasets for faster training
print("Creating stratified samples...")
train_small = stratified_sample(train_df, min(REDUCED_TRAIN, len(train_df)))
dev_small = stratified_sample(dev_df, min(REDUCED_DEV, len(dev_df)))
test_small = stratified_sample(test_df, min(REDUCED_TEST, len(test_df))) if test_df is not None else None

print(f"Sample sizes - Training: {len(train_small)}, Validation: {len(dev_small)}, Test: {len(test_small) if test_small is not None else 'Not available'}")

# Create label mappings (convert labels to numerical IDs)
if train_small['label'].dtype == object:
    # String labels - create mapping
    unique_labels = sorted(train_small['label'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Apply mapping to all datasets
    train_small['label_id'] = train_small['label'].map(label2id)
    dev_small['label_id'] = dev_small['label'].map(label2id)
    if test_small is not None:
        test_small['label_id'] = test_small['label'].map(label2id)
else:
    # Numerical labels - use as is
    unique_labels = sorted(train_small['label'].unique())
    label2id = {str(label): int(label) for label in unique_labels}
    id2label = {int(label): str(label) for label in unique_labels}
    train_small['label_id'] = train_small['label'].astype(int)
    dev_small['label_id'] = dev_small['label'].astype(int)
    if test_small is not None:
        test_small['label_id'] = test_small['label'].astype(int)

print("Label mapping:", label2id)
print("Number of classes:", len(label2id))

# Check for empty datasets
if len(train_small) == 0:
    raise ValueError("Training dataset is empty after sampling")
if len(dev_small) == 0:
    raise ValueError("Validation dataset is empty after sampling")

# Convert pandas DataFrames to Hugging Face datasets
hf_train = Dataset.from_pandas(train_small[['text', 'label_id']].rename(columns={'label_id': 'labels'}))
hf_dev = Dataset.from_pandas(dev_small[['text', 'label_id']].rename(columns={'label_id': 'labels'}))
hf_test = Dataset.from_pandas(test_small[['text', 'label_id']].rename(columns={'label_id': 'labels'})) if test_small is not None else None

# Initialize tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Tokenization function
def tokenize_function(batch):
    """
    Tokenize text data for model input
    """
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=MAX_LEN)

# Apply tokenization to datasets
print("Tokenizing datasets...")
hf_train = hf_train.map(tokenize_function, batched=True)
hf_dev = hf_dev.map(tokenize_function, batched=True)
if hf_test is not None:
    hf_test = hf_test.map(tokenize_function, batched=True)

# Set dataset format for PyTorch
hf_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
hf_dev.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
if hf_test is not None:
    hf_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Function to compute evaluation metrics
def compute_metrics(prediction):
    """
    Compute accuracy and F1 scores for model evaluation
    """
    predictions = np.argmax(prediction.predictions, axis=1)
    labels = prediction.label_ids
    
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./training_output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="no",
    report_to=None
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_dev,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Start training
print("Starting model training...")
print(f"Training samples: {len(hf_train)}, Validation samples: {len(hf_dev)}")
training_history = trainer.train()

# Evaluate on validation set
print("Evaluating on validation set...")
validation_metrics = trainer.evaluate()
print("Validation Metrics:")
print(json.dumps(validation_metrics, indent=2))

# Evaluate on test set if available
if hf_test is not None:
    print("Evaluating on test set...")
    test_predictions = trainer.predict(hf_test)
    test_metrics = test_predictions.metrics
    print("Test Metrics:")
    print(json.dumps(test_metrics, indent=2))
    
    # Detailed classification report for test set
    test_true_labels = test_predictions.label_ids
    test_pred_labels = np.argmax(test_predictions.predictions, axis=1)
    
    print("Detailed Classification Report (Test Set):")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(test_true_labels, test_pred_labels, target_names=target_names))

# Generate and display confusion matrix for validation set
print("Generating confusion matrix...")
validation_predictions = trainer.predict(hf_dev)
true_labels = validation_predictions.label_ids
predicted_labels = np.argmax(validation_predictions.predictions, axis=1)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix (Validation Set):")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Set")
plt.colorbar()

class_names = [id2label[i] for i in range(len(id2label))]
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add text annotations to confusion matrix
thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Training summary
print("Training completed successfully!")
print(f"Validation Accuracy: {validation_metrics['eval_accuracy']:.4f}")
print(f"Validation F1 Score (Weighted): {validation_metrics['eval_f1_weighted']:.4f}")
print(f"Validation F1 Score (Macro): {validation_metrics['eval_f1_macro']:.4f}")

# Save the trained model
print("Saving model...")
trainer.save_model("./sentiment_analysis_model")
print("Model saved to: ./sentiment_analysis_model")

print("Script execution completed!")
