# Machine-Learning-Classification

# CIFAR-10 Image Classification Coursework

This MATLAB-based project applies multiple machine learning models to classify images from the CIFAR-10 dataset. It demonstrates a full machine learning pipeline including data preprocessing, model training, and performance evaluation.

---

##  Overview

We use a subset of the CIFAR-10 dataset (focusing on 3 classes: **horse**, **truck**, **deer**) and evaluate four classification methods:

-  Custom K-Nearest Neighbors (KNN) with:
  - Euclidean Distance
  - Cosine Similarity
-  Decision Tree
-  Support Vector Machine (SVM)

Each classifier is tested for accuracy and runtime performance. Visual results include confusion matrices and sample input images.

---

##  Algorithms Used

| Model           | Description                           |
|-----------------|---------------------------------------|
| KNN (Euclidean) | Custom-coded using Euclidean distance |
| KNN (Cosine)    | Custom-coded using Cosine similarity  |
| Decision Tree   | Implemented using `fitctree`          |
| SVM             | Implemented using `fitcecoc`          |

---

## 🛠 Project Files

- `Script.m` – Main script for data loading, training, testing, and visualization
- `cifar-10-data.mat` – CIFAR-10 dataset (external file assumed available)
- `cw1.mat` – Output containing performance metrics and confusion matrices

---

##  How It Works

1. **Load & Normalize** image data from CIFAR-10
2. **Filter** to only include selected classes
3. **Split** into training and testing sets (50/50)
4. **Reshape** image data into 1D vectors for model input
5. **Train & Evaluate** models
6. **Output** accuracy, runtime, and confusion matrices

## CIFAR-10-DATA.MAT Removal due to size limitation
- cifar-10-data.mat exceeded github's size limit of 100MB, which is why is has been removed from the initial commit
- Should be pushed as soon as possible, omititon is due to the desire to push the project to git as soon as possible
---


