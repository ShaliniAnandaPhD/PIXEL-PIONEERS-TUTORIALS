# File name: jax_bert_sentiment_analysis.py

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
from flax.training.common_utils import shard
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = FlaxAutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Define the sentiment labels
label_map = {"NEGATIVE": 0, "POSITIVE": 1}

# Define the data preprocessing function
def preprocess_data(examples):
    """
    Preprocess the dataset by tokenizing the text and mapping sentiment labels to integers.
    
    Parameters:
    examples (dict): A dictionary containing text and sentiment labels.
    
    Returns:
    dict: A dictionary with tokenized inputs and integer labels.
    """
    inputs = tokenizer(examples["text"], padding=True, truncation=True, max_length=128)
    inputs["label"] = [label_map[label] for label in examples["sentiment"]]
    return inputs

# Load and preprocess the dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"].map(preprocess_data, batched=True, remove_columns=["text", "sentiment"])
val_dataset = dataset["test"].map(preprocess_data, batched=True, remove_columns=["text", "sentiment"])

# Define the training loop
@jax.pmap
def train_step(model, inputs, labels):
    """
    Perform a single training step.

    Parameters:
    model: The BERT model.
    inputs: Tokenized input data.
    labels: Corresponding labels for the input data.

    Returns:
    The updated model after applying the gradients.
    """
    def loss_fn(params):
        logits = model(params=params, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        loss = jnp.mean(jax.vmap(jax.nn.sparse_softmax_cross_entropy_with_logits)(logits, labels))
        return loss
    
    gradients = jax.grad(loss_fn)(model.params)
    model = model.apply_gradients(grads=gradients)
    return model

# Define the evaluation metrics
def compute_metrics(logits, labels):
    """
    Compute evaluation metrics for the model.

    Parameters:
    logits: Predicted logits from the model.
    labels: True labels.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define the training and evaluation loop
def train_and_evaluate(model, train_dataset, val_dataset, num_epochs, batch_size):
    """
    Train and evaluate the BERT model.

    Parameters:
    model: The BERT model.
    train_dataset: The training dataset.
    val_dataset: The validation dataset.
    num_epochs: Number of training epochs.
    batch_size: Size of each training batch.

    Returns:
    The trained model.
    """
    for epoch in range(num_epochs):
        # Training loop
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            inputs = shard(batch)
            labels = shard(batch["label"])
            model = train_step(model, inputs, labels)
        
        # Evaluation loop
        val_logits = []
        val_labels = []
        for batch in val_dataset:
            inputs = shard(batch)
            labels = batch["label"]
            logits = model(params=model.params, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            val_logits.append(logits)
            val_labels.extend(labels)
        
        val_logits = jnp.concatenate(val_logits, axis=0)
        val_labels = jnp.array(val_labels)
        metrics = compute_metrics(val_logits, val_labels)
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    return model

# Set hyperparameters
num_epochs = 3
batch_size = 16

# Train and evaluate the model
model = train_and_evaluate(model, train_dataset, val_dataset, num_epochs, batch_size)

# Save the fine-tuned model
model.save_pretrained("sentiment_analysis_bert")

# Possible Errors and Solutions:
# 1. ImportError: cannot import name 'FlaxAutoModelForSequenceClassification'.
#    Solution: Ensure you have the correct version of the `transformers` library installed. Use `pip install transformers --upgrade`.

# 2. RuntimeError: CUDA out of memory.
#    Solution: Reduce the batch size or use gradient accumulation to fit the model into available GPU memory.

# 3. KeyError: 'sentiment'.
#    Solution: Ensure the dataset has the correct column names. Verify the dataset structure using `dataset.column_names`.

# 4. TypeError: 'NoneType' object is not subscriptable.
#    Solution: Check that the tokenizer and model are correctly loaded. Verify paths and model names.

