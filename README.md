# Facial Recognition System: Eigenfaces & ANN

## Overview

This project implements a complete facial recognition system from scratch using **Principal Component Analysis (PCA)** for feature extraction and a **Multi-Layer Perceptron (MLP) Artificial Neural Network** for classification.

Rather than relying on pre-built deep learning black boxes, this project mathematically constructs the Eigenfaces algorithm, utilizing the Surrogate Covariance trick to bypass computational bottlenecks when processing high-dimensional image matrices.

## Key Features

* **Automated Data Processing:** Dynamically loads, resizes, and flattens grayscale face images into mean-zero column vectors.
* **Dimensionality Reduction:** Computes the surrogate covariance matrix and extracts the top $k$ eigenvectors (Eigenfaces) using optimized Scipy sparse linear algebra solvers.
* **ANN Classification:** Projects mean-aligned faces onto the feature space to generate unique mathematical signatures, training a Backpropagation ANN on a 60/40 data split.
* **Imposter Detection:** Implements a security threshold using Euclidean distance to confidently reject out-of-distribution inputs (anomalies/imposters).

## Results & Optimization

A comparative study was conducted to find the optimal number of Principal Components ($k$). The model achieved peak classification efficiency at **$k = 75$**.
Additionally, the system successfully rejected unverified imposter data with an astronomical Euclidean distance score, proving the system's strict boundary recognition.

## Technology Stack

* **Language:** Python
* **Libraries:** Scikit-Learn (ANN, Splitting), SciPy (Eigendecomposition), OpenCV (Image Processing), NumPy & Pandas (Matrix Operations), Matplotlib (Data Visualization).
