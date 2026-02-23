# Neural Network for Diabetes Diagnosis

## Course: Deep Learning for Medicine
**Student Name:** Mahmoud Mohamed Abdelfattah
**Student ID:** 4220142
**Date:** February 2026

---

## Project Overview

This project applies a neural network binary classification pipeline to the **Pima Indians Diabetes Dataset** to predict whether a patient has diabetes based on 8 clinical measurements. The complete workflow includes data loading, exploration, preprocessing, model building, training with early stopping, evaluation, and architecture comparison.

## Dataset

- **Name:** Pima Indians Diabetes Database
- **Source:** [Kaggle - UCI Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download)
- **Samples:** 768 patients
- **Features:** 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Target:** Outcome (0 = No Diabetes, 1 = Diabetes)

## Approach

1. **Data Preprocessing:** Replaced invalid zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI with median values. Applied StandardScaler for feature normalization.
2. **Train/Test Split:** 80% training, 20% testing with stratified sampling.
3. **Model Architecture:** Sequential NN — Input(8) → Dense(32, ReLU) → Dense(16, ReLU) → Output(1, Sigmoid)
4. **Training:** Adam optimizer (lr=0.001), binary crossentropy loss, EarlyStopping (patience=5), max 100 epochs, batch size 32.
5. **Evaluation:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. **Architecture Comparison:** Built and compared a second model (V2) with modified architecture.

## Results

- **Expected Test Accuracy:** 70–75%
- **Key Challenge:** Class imbalance (~65% no diabetes vs ~35% diabetes)
- **Medical Priority:** Maximizing recall for the diabetes class to minimize missed diagnoses

## Repository Structure

```
neural-network-diabetes/
├── pima_diabetes_nn.ipynb      # Complete notebook with code and results
├── diabetes.csv                # Dataset file
├── README.md                   # This file
└── results/
    ├── training_curves.png
    ├── confusion_matrix.png
    └── metrics_summary.txt
```

## How to Run

1. Ensure Python 3.x with TensorFlow, scikit-learn, pandas, matplotlib, and seaborn installed.
2. Place `diabetes.csv` in the same directory as the notebook.
3. Open `pima_diabetes_nn.ipynb` and run all cells sequentially.

## Tools & Libraries

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
