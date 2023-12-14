import torch
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

def load_iris_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def split_data(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def train_and_evaluate(features_train, labels_train, features_test, labels_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(features_train, labels_train).predict(features_test)

    print("Number of mislabeled points out of a total %d points : %d" % (features_test.shape[0], (labels_test != y_pred).sum()))

   
    print("Confusion Matrix:\n", confusion_matrix(labels_test, y_pred))
    print("\nClassification Report:\n", classification_report(labels_test, y_pred))

    scores = cross_val_score(gnb, features_train, labels_train, cv=5)
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))

  
    plot_decision_regions(features_train, labels_train, clf=gnb, legend=2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Regions of Gaussian Naive Bayes')
    plt.show()

def show_seaborn_histogram(data):
    flat_data = data.flatten()
    sns.histplot(flat_data, bins=50, kde=False)
    plt.title('Seaborn Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def create_tkinter_gui():
    root = tk.Tk()
    root.title("Advanced Iris Classification")

    iris_features, iris_labels = load_iris_data()
    features_train, features_test, labels_train, labels_test = split_data(iris_features, iris_labels)

    train_and_evaluate(features_train, labels_train, features_test, labels_test)

    show_seaborn_histogram(iris_features)

    root.mainloop()

