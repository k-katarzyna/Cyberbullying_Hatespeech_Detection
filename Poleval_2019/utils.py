import os
import random
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

__all__ = ["load_data_and_labels_from_zip",
           "display_samples",
           "display_class_distribution",
           "model_results"]


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
    
    f1_micro = f1_score(test_labels, y_pred, average="micro")
    f1_macro = f1_score(test_labels, y_pred, average="macro")

    return {
        "model": pipeline.steps[1][1].__class__.__name__,
        "microF": f1_micro,
        "macroF": f1_macro
    }