# sentiment_analysis_autosklearn_tutorial.py

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import autosklearn.classification
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Step 1: Load and preprocess the dataset
def load_and_preprocess_data():
    """
    Load and preprocess the 20 Newsgroups dataset for sentiment analysis.
    
    Returns:
        tuple: Preprocessed data and labels.
    """
    try:
        logger.info("Loading 20 Newsgroups dataset...")
        newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'])
        newsgroups_test = fetch_20newsgroups(subset='test', categories=['alt.atheism', 'sci.space'])
        
        # Combine the training and testing data
        data = np.concatenate((newsgroups_train.data, newsgroups_test.data), axis=0)
        labels = np.concatenate((newsgroups_train.target, newsgroups_test.target), axis=0)
        
        # Preprocess the text data using CountVectorizer
        logger.info("Preprocessing the text data...")
        vectorizer = CountVectorizer(stop_words='english')
        data = vectorizer.fit_transform(data)
        
        return data, labels
    
    except Exception as e:
        logger.error(f"Error occurred while loading and preprocessing the data: {str(e)}")
        raise

# Step 2: Split the dataset into training and testing sets
def split_dataset(data, labels, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        data (numpy.ndarray): Preprocessed data.
        labels (numpy.ndarray): Corresponding labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: Training and testing data and labels.
    """
    try:
        logger.info("Splitting the dataset into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    except ValueError as ve:
        logger.error(f"ValueError occurred while splitting the dataset: {str(ve)}")
        raise
    
    except Exception as e:
        logger.error(f"Error occurred while splitting the dataset: {str(e)}")
        raise

# Step 3: Train the sentiment analysis model using auto-sklearn
def train_model(X_train, y_train, time_limit=600):
    """
    Train the sentiment analysis model using auto-sklearn.
    
    Args:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        time_limit (int): Time limit for the AutoML optimization in seconds.
        
    Returns:
        autosklearn.classification.AutoSklearnClassifier: Trained auto-sklearn classifier.
    """
    try:
        logger.info("Training the sentiment analysis model using auto-sklearn...")
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_limit,
            per_run_time_limit=30,
            memory_limit=4096,
            n_jobs=-1,
            random_state=42
        )
        
        automl.fit(X_train, y_train)
        
        return automl
    
    except ValueError as ve:
        logger.error(f"ValueError occurred while training the model: {str(ve)}")
        raise
    
    except MemoryError as me:
        logger.error(f"MemoryError occurred while training the model. Consider increasing the memory limit or reducing the dataset size.")
        raise
    
    except TimeoutError as te:
        logger.warning(f"TimeoutError occurred while training the model. Consider increasing the time limit or optimizing the dataset.")
        raise
    
    except Exception as e:
        logger.error(f"Error occurred while training the model: {str(e)}")
        raise

# Step 4: Evaluate the trained model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained sentiment analysis model.
    
    Args:
        model (autosklearn.classification.AutoSklearnClassifier): Trained auto-sklearn classifier.
        X_test (numpy.ndarray): Testing data.
        y_test (numpy.ndarray): Testing labels.
        
    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, F1 score).
    """
    try:
        logger.info("Evaluating the trained sentiment analysis model...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error occurred while evaluating the model: {str(e)}")
        raise

# Step 5: Perform sentiment analysis on new text data
def predict_sentiment(model, text_data):
    """
    Perform sentiment analysis on new text data.
    
    Args:
        model (autosklearn.classification.AutoSklearnClassifier): Trained auto-sklearn classifier.
        text_data (list): List of text data to analyze.
        
    Returns:
        numpy.ndarray: Predicted sentiment labels.
    """
    try:
        logger.info("Performing sentiment analysis on new text data...")
        vectorizer = CountVectorizer(stop_words='english')
        X_new = vectorizer.fit_transform(text_data)
        
        y_pred = model.predict(X_new)
        
        return y_pred
    
    except Exception as e:
        logger.error(f"Error occurred while performing sentiment analysis: {str(e)}")
        raise

# Step 6: Save the trained model
def save_model(model, file_path):
    """
    Save the trained sentiment analysis model to a file.
    
    Args:
        model (autosklearn.classification.AutoSklearnClassifier): Trained auto-sklearn classifier.
        file_path (str): File path to save the model.
    """
    try:
        logger.info(f"Saving the trained model to {file_path}...")
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        
        logger.info("Model saved successfully.")
    
    except IOError as ioe:
        logger.error(f"IOError occurred while saving the model: {str(ioe)}")
        raise
    
    except Exception as e:
        logger.error(f"Error occurred while saving the model: {str(e)}")
        raise

# Step 7: Load the saved model
def load_model(file_path):
    """
    Load the saved sentiment analysis model from a file.
    
    Args:
        file_path (str): File path to load the model.
        
    Returns:
        autosklearn.classification.AutoSklearnClassifier: Loaded auto-sklearn classifier.
    """
    try:
        logger.info(f"Loading the model from {file_path}...")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        
        logger.info("Model loaded successfully.")
        return model
    
    except FileNotFoundError as fnfe:
        logger.error(f"FileNotFoundError occurred while loading the model: {str(fnfe)}")
        raise
    
    except IOError as ioe:
        logger.error(f"IOError occurred while loading the model: {str(ioe)}")
        raise
    
    except Exception as e:
        logger.error(f"Error occurred while loading the model: {str(e)}")
        raise

# Main function
def main():
    try:
        # Load and preprocess the dataset
        data, labels = load_and_preprocess_data()
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = split_dataset(data, labels)
        
        # Train the sentiment analysis model using auto-sklearn
        model = train_model(X_train, y_train)
        
        # Evaluate the trained model
        metrics = evaluate_model(model, X_test, y_test)
        logger.info(f"Evaluation Metrics: {metrics}")
        
        # Perform sentiment analysis on new text data
        new_text_data = [
            "This movie was amazing! The acting was brilliant.",
            "I didn't enjoy the book. It was quite disappointing.",
            "The product worked well, but the customer service was terrible."
        ]
        sentiments = predict_sentiment(model, new_text_data)
        logger.info(f"Predicted Sentiments: {sentiments}")
        
        # Save the trained model
        model_file_path = 'sentiment_analysis_model.pkl'
        save_model(model, model_file_path)
        
        # Load the saved model
        loaded_model = load_model(model_file_path)
        
        logger.info("Sentiment Analysis Tutorial completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred during the sentiment analysis tutorial: {str(e)}")

if __name__ == '__main__':
    main()
