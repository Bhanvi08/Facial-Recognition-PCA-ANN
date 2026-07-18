# Facial Recognition System | Eigenfaces & ANN 👤🧠
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![ML-Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Math-Library](https://img.shields.io/badge/Mathematics-SciPy-lightblue.svg?style=flat-square&logo=scipy)](https://scipy.org/)

## 📝 Project Overview
This project implements a complete facial recognition system from scratch using **Principal Component Analysis (PCA)** for feature extraction and a **Multi-Layer Perceptron (MLP) Artificial Neural Network** for classification. 

Rather than relying on pre-built deep learning black boxes, this project mathematically constructs the Eigenfaces algorithm, utilizing the Surrogate Covariance trick to bypass computational bottlenecks when processing high-dimensional image matrices.

---

## 🚀 Key Technical Features
*   **Automated Data Processing:** Dynamically loads, resizes, and flattens grayscale face images into mean-zero column vectors.
*   **Dimensionality Reduction:** Computes the surrogate covariance matrix and extracts the top $k$ eigenvectors (Eigenfaces) using optimized SciPy sparse linear algebra solvers.
*   **ANN Classification:** Projects mean-aligned faces onto the feature space to generate unique mathematical signatures, training a Backpropagation ANN on a 60/40 data split.
*   **Imposter Detection:** Implements a strict security threshold using Euclidean distance to confidently reject out-of-distribution inputs (anomalies/imposters).

---

## 📊 Results & Optimization
A comparative study was conducted to find the optimal number of Principal Components ($k$). The model achieved peak classification efficiency at **$k = 75$**. 

Additionally, the system successfully rejected unverified imposter data with an astronomical Euclidean distance score, mathematically proving the system's strict boundary recognition and robust anomaly detection.

---

## 🛠️ Technology Stack
*   **Language:** Python
*   **Libraries:** 
    *   `scikit-learn` (ANN Classification, Train/Test Splitting)
    *   `SciPy` (Eigendecomposition & Sparse Matrix Math)
    *   `OpenCV` (Image Processing)
    *   `NumPy` & `Pandas` (High-Dimensional Matrix Operations)
    *   `Matplotlib` (Data Visualization)

Author: Bhanvi Singh Chauhan
