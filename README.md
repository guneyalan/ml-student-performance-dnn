# Student Performance Deep Neural Network

## Project Description

This project uses a Deep Neural Network to predict whether a student is a high performer based on different educational and personal factors.

The model predicts whether a student is likely to achieve a high exam score.

---

## Technologies Used

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- Matplotlib

---

## Dataset

Dataset used:
Student Performance Factors (Kaggle)

The dataset contains features such as:

- Hours Studied
- Attendance
- Sleep Hours
- Motivation Level
- Previous Scores
- Internet Access
- Family Income
- Teacher Quality

---

## Machine Learning Concepts

This project demonstrates:

- Deep Neural Networks
- Hidden Layers
- Forward Propagation
- Backpropagation
- Activation Functions
- Data Preprocessing
- Model Evaluation

---

## Neural Network Architecture

Input Layer:
Student data features

Hidden Layers:
- Dense(16, activation="relu")
- Dense(8, activation="relu")

Output Layer:
- Dense(1, activation="sigmoid")

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report

Example result:
- Test Accuracy ≈ 91%

---

## How To Run

```bash
pip install -r requirements.txt
python main.py