import os
import pickle
import random
import zipfile
import re

from morfeusz2 import Morfeusz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, ConfusionMatrixDisplay

SEED = 42
CV_SCHEME = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

__all__ = ["SEED",
           "load_data_and_labels_from_zip",
           "display_samples",
           "display_class_distribution",
           "model_results",
           "LemmaTransformer",
           "optimize_models",
           "display_confusion_matrix"]


def load_data_and_labels_from_zip(zip_file_path, data_txt_file, labels_txt_file):
    """
    Load text data and labels from text files within a zip archive.

    Args:
        zip_file_path (str): Path to the zip file.
        data_txt_file (str): Name or path of the text file containing data.
        labels_txt_file (str): Name or path of the text file containing labels (numbers).

    Returns:
        data (list): List containing text data.
        labels (list): List containing labels (numbers).
    """
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"Zip file '{zip_file_path}' does not exist.")

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        if data_txt_file not in zip_ref.namelist():
            raise FileNotFoundError(f"File '{data_txt_file}' does not exist in '{zip_file_path}'.")
        if labels_txt_file not in zip_ref.namelist():
            raise FileNotFoundError(f"File '{labels_txt_file}' does not exist in '{zip_file_path}'.")

        with zip_ref.open(data_txt_file) as data_file:
            data = data_file.read().decode("utf-8").splitlines()

        with zip_ref.open(labels_txt_file) as labels_file:
            labels = labels_file.read().decode("utf-8").splitlines()

        labels = [int(label) for label in labels]

    return data, labels


def display_samples(data, labels, n_samples=1):
    """
    Display random data samples for each label.
    
    Args:
        data (list of strings): Test data for analysis.
        labels (list of ints): Corresponding labels.
        n_samples (int, default=1): Number of samples of each class to display.
    """
    unique_labels = set(labels)
    for label in sorted(unique_labels):
        label_indices = [i for i, l in enumerate(labels) if l == label]
        print(f"Examples with label {label}:")
        for _ in range(min(n_samples, len(label_indices))):
            index = random.choice(label_indices)
            print(f"Text: {data[index]}")
        print()


def calculate_distribution(labels):
    """Calculate the percentage for each class."""
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    return percentages
    

def display_class_distribution(train_labels, test_labels):
    """
    Display class distributions for train and test labels.
    
    Args:
        train_labels (list of ints): Train labels.
        train_labels (list of ints): Test labels.
    """
    pie_labels = ["0 (non-harfmul)", "1 (cyberbullying)", "2 (hate-speech)"]
    colors = ['#ff9999', '#ffd966', '#b3d9ff']
    train_percentages = calculate_distribution(train_labels)
    test_percentages = calculate_distribution(test_labels)

    pie_dict = dict(labels=pie_labels,
                    autopct="%1.1f",
                    startangle=150,
                    labeldistance=1.1,
                    colors=colors)

    plt.figure(figsize=(10, 6))
    plt.subplot(221)
    plt.pie(train_percentages, **pie_dict)
    plt.title("Train label distribution [%]")
    plt.subplot(222)
    plt.pie(test_percentages, **pie_dict)
    plt.title("Test label distribution [%]")
    plt.tight_layout()
    plt.show()


def model_results(pipeline, train_data, train_labels, test_data, test_labels):
    """Fit, predict and calculate pipeline scores."""

    pipeline.fit(train_data, train_labels)
    y_pred = pipeline.predict(test_data)
    
    f1_macro = f1_score(test_labels, y_pred, average="macro")
    f1_micro = f1_score(test_labels, y_pred, average="micro")
    bal_acc = balanced_accuracy_score(test_labels, y_pred)

    return {
        "model": pipeline.steps[1][1].__class__.__name__,
        "macroF": f1_macro,
        "microF": f1_micro,
        "balanced_accuracy": bal_acc
    }


class LemmaTransformer(TransformerMixin):
    """
    LemmaTransformer is a class for lemmatizing text data.

    Parameters:
    - token_pattern: str, optional
        Regular expression pattern to identify tokens in the text. Defaults to '(?u)\\b\\w\\w+\\b'.

    Methods:
    - _process_token(token): 
        Processes a single token using Morfeusz analyzer and returns its lemma.
    - _lemmatize_tweet(tweet):
        Lemmatizes a tweet by replacing each token with its lemma (other elements unchanged).
    - fit(X, y=None):
        Initializes a semantic text analyser Morfeusz object.
    - transform(X):
        Transforms input data by lemmatizing each tweet. Returns a list of lemmatized tweets.

    Note:
    Morfeusz initialization is performed in the `fit` method. This approach enables parallel
    computations to be executed without serialization issues. As a result, this transformer
    can be used in parallelized operations without encountering the 'TypeError: cannot pickle
    'SwigPyObject' object' error, unlike initializing the object in the `__init__` method
    or externally.
    """    
    def __init__(self, token_pattern="(?u)\\b\\w\\w+\\b"):
        self.token_pattern = token_pattern
    
    def _process_token(self, token):
        """Process a single token using Morfeusz analyzer and returns its lemma."""
        analyse = self.morfeusz_.analyse(token)
        pre_lemma = analyse[0][2][1]
        lemma = pre_lemma.split(":")[0]
        return lemma
    
    def _lemmatize_tweet(self, tweet):
        """Lemmatize a tweet by replacing each token with its lemma."""
        tokens = re.findall(self.token_pattern, tweet)
        for token in tokens:
            lemma = self._process_token(token)
            tweet = tweet.replace(token, lemma)
        return tweet
    
    def fit(self, X, y=None):
        """Initialize a Morfeusz object."""
        self.morfeusz_ = Morfeusz()
        return self
    
    def transform(self, X):
        """Transform input data by lemmatizing each tweet. Returns a list of lemmatized tweets."""
        lemmatized_tweets = []
        for tweet in X:
            lemmatized_tweet = self._lemmatize_tweet(tweet)
            lemmatized_tweets.append(lemmatized_tweet)
        return lemmatized_tweets


def get_metrics(best_idx, best_score, cv_results, name, test, y_pred, test_labels):
    """
    Calculate evaluation metrics for a model.

    Args:
        best_idx (int): Index of the best performing model in cv_results.
        best_score (float): Best score achieved during cross-validation.
        cv_results (dict): Dictionary containing results of cross-validation.
        name (str): Name of the model.
        test (str): Method of class balancing.
        y_pred (array-like): Predicted labels on the test data.
        test_labels (array-like): True labels of the test data.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    cv_f1_micro = cv_results["mean_test_f1_micro"][best_idx]
    cv_balanced_accuracy = cv_results["mean_test_balanced_accuracy"][best_idx]
    time = np.round((cv_results["mean_fit_time"]
                     + cv_results["mean_score_time"])[best_idx], 2)
    
    test_f1_macro = f1_score(test_labels, y_pred, average="macro")
    test_f1_micro = f1_score(test_labels, y_pred, average="micro")
    test_balanced_accuracy = balanced_accuracy_score(test_labels, y_pred)

    dict_result = {
        "model": name,
        "test": test,
        "cv_macroF": best_score,
        "cv_microF": cv_f1_micro,
        "cv_balanced_acc": cv_balanced_accuracy,
        "test_macroF": test_f1_macro,
        "test_microF": test_f1_micro,
        "test_balanced_acc": test_balanced_accuracy,
        "time [s]": time
    }

    return dict_result


def save_params(params, test, name):
    """Save best estimator's params as pkl file."""
    save_params_path = os.path.join("results", "params_artifacts", test, f"{name}.pkl")
    with open(save_params_path, "wb") as f:
        pickle.dump(params, f)


def optimize_models(names, model_grids, preprocess_grid, pipeline,
                    train_data, train_labels, test_data, test_labels,
                    test, n_iter=200, refit="f1_macro"):
    """
    Optimize multiple models using RandomizedSearchCV and evaluate them on test data.

    Args:
        names (list): List of model names.
        model_grids (list): List of dictionaries containing hyperparameter grids for each model.
        preprocess_grid (list of 2 dicts): Hyperparameter grid for preprocessing steps.
        pipeline (object): Pipeline object for model training.
        train_data (array-like): Training data.
        train_labels (array-like): True labels of the training data.
        test_data (array-like): Test data.
        test_labels (array-like): True labels of the test data.
        test (str): A method of class balancing.
        n_iter (int, optional): Number of parameter settings that are sampled. Defaults to 200.
        refit (str, optional): Metric used to choose the best model. Defaults to "f1_macro".

    Returns:
        DataFrame: DataFrame containing evaluation metrics for each model.
    """
    results = []

    for name, model_grid in zip(names, model_grids):
        
        param_grid = {
            **preprocess_grid,
            **model_grid
        }
        
        optimizer = RandomizedSearchCV(pipeline,
                                       param_grid,
                                       cv=CV_SCHEME,
                                       scoring=["f1_macro", "f1_micro",
                                                "balanced_accuracy"],
                                       refit=refit,
                                       n_iter=n_iter,
                                       n_jobs=-1,
                                       error_score="raise"
                                      )
        optimizer.fit(train_data, train_labels)
        y_pred = optimizer.best_estimator_.predict(test_data)

        save_params(optimizer.best_params_, test, name)
        
        scores = get_metrics(optimizer.best_index_,
                             optimizer.best_score_,
                             optimizer.cv_results_,
                             name,
                             test,
                             y_pred,
                             test_labels)
        results.append(scores)
        
    return pd.DataFrame(results)


def display_confusion_matrix(estimator, X_test, y_test, cmap="summer"):
    """Display minimalistic confusion matrix from fitted estimator"""
    fig, ax = plt.subplots(figsize=(3, 3))
    ConfusionMatrixDisplay.from_estimator(estimator,
                                          X_test, y_test,
                                          ax=ax,
                                          cmap=cmap,
                                          colorbar=False);