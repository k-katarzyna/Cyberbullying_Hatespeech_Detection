import os
import random
import zipfile

import numpy as np
import matplotlib.pyplot as plt


def load_data_and_labels_from_zip(zip_file_path, data_txt_file_name, labels_txt_file_name):
    """
    Load text data and labels from text files within a zip archive.

    Args:
    - zip_file_path (str): Path to the zip file.
    - data_txt_file_name (str): Name of the text file containing data.
    - labels_txt_file_name (str): Name of the text file containing labels (numbers).

    Returns:
    - data (list): List containing text data.
    - labels (list): List containing labels (numbers).
    """
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"Zip file '{zip_file_path}' does not exist.")

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        if data_txt_file_name not in zip_ref.namelist():
            raise FileNotFoundError(f"File '{data_txt_file_name}' does not exist in the archive '{zip_file_path}'.")
        if labels_txt_file_name not in zip_ref.namelist():
            raise FileNotFoundError(f"File '{labels_txt_file_name}' does not exist in the archive '{zip_file_path}'.")

        with zip_ref.open(data_txt_file_name) as data_file:
            data = data_file.read().decode("utf-8").splitlines()

        with zip_ref.open(labels_txt_file_name) as labels_file:
            labels = labels_file.read().decode("utf-8").splitlines()

        labels = [int(label) for label in labels]

    return data, labels


def display_samples(data, labels, n_samples=1):
    
    unique_labels = set(labels)
    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        print(f"Examples with label {label}:")
        for _ in range(min(n_samples, len(label_indices))):
            index = random.choice(label_indices)
            print(f"Text: {data[index]}")
        print()


def display_class_distribution(labels):
    
    counts = np.bincount(labels)
    percentages = counts / len(labels) * 100
    pie_labels = ["0 (non-harfmul)", "1 (cyberbullying)", "2 (hate-speech)"]
    colors = ['#ff9999', '#ffd966', '#b3d9ff']
    
    plt.figure(figsize=(4, 4))
    plt.pie(percentages, labels=pie_labels,
            autopct="%1.1f%%", startangle=150,
            labeldistance=1.1, colors=colors)
    plt.title("Percentage of each label")
    plt.show()