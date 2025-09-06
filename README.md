# Heart Disease Project
Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset Final project @ Sprints x Microsoft Summer Camp - AI and Machine Learning.
 
## Table of Contents
  
- [Instructions](#Instructions)
- [Project-Structure](#Project-Structure)
- [Requirements](#requirements)
- [Dataset](#Dataset)
- [Notebooks](#Notebooks)
- [Models](#Models)
- [Results](#Results)
- [UI](#UI)
  
## Instructions
1. Objectives:
- Perform Data Preprocessing & Cleaning (handle missing values, encoding, scaling).
- Apply Dimensionality Reduction (PCA) to retain essential features.
- Implement Feature Selection using statistical methods and ML-based techniques.
- Train Supervised Learning Models (Logistic Regression, Decision Trees, Random Forest, SVM) for classification.
- Apply Unsupervised Learning (K-Means, Hierarchical Clustering) for pattern discovery.
- Optimize Models using Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV).
- Deploy a Streamlit UI for real-time user interaction. [Bonus]
- Host the application using Ngrok [Bonus] and upload the project to GitHub for accessibility.
2. Tools to be Used:
- Programming Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, TensorFlow/Keras (optional).
- Dimensionality Reduction & Feature Selection: PCA, RFE, Chi-Square Test.
- Supervised Models: Logistic Regression, Decision Trees, Random Forest, SVM.
- Unsupervised Models: K-Means, Hierarchical Clustering.
- Model Optimization: GridSearchCV, RandomizedSearchCV.
- Deployment Tools: Streamlit [Bonus], Ngrok [Bonus], GitHub.

## Project-Structure
```
Heart_Disease_Project/
│── data/
│ ├── heart_disease.csv
| ├── clean_data.csv
| ├── features_selected_rfe.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ ├── 06_hyperparameter_tuning.ipynb
│── models/
│ ├── model_pipeline.pkl
│── ui/
│ ├── app.py (Streamlit UI)
│── results/
│ ├── no_risk.png
| ├── high_risk.png
│── README.md
│── requirements.txt
│── .gitignore

```
## Requirements

```bash
pip install requirements.txt
```

## Dataset
- [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Notebooks
- [Notebooks](https://github.com/Asma-Nasr/Heart-Disease-Project/tree/main/notebooks)

## Models
- [Models](https://github.com/Asma-Nasr/Heart-Disease-Project/tree/main/models)

## Results
![High Risk](https://github.com/Asma-Nasr/Heart-Disease-Project/blob/main/results/high_risk.png)

![No Risk](https://github.com/Asma-Nasr/Heart-Disease-Project/blob/main/results/no_risk.png)

## UI
- [UI](https://github.com/Asma-Nasr/Heart-Disease-Project/tree/main/ui)
