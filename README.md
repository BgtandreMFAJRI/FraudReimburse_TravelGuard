# ReimburseGuard

## Fraud Detection System for Business Travel Reimbursement Using Decision Tree and Risk Scoring

ReimburseGuard is a Machine Learning-based fraud detection system designed to identify suspicious business travel reimbursement claims. The system utilizes a Decision Tree algorithm combined with a Risk Scoring approach to classify reimbursement submissions into Low Risk, Medium Risk, and High Risk categories.

This project was developed as a final assignment for the Machine Learning course at Politeknik Negeri Indramayu.

---

## Overview

Manual verification of reimbursement claims is often time-consuming and vulnerable to fraud, such as:

- Inflated claim amounts
- Incomplete supporting documents
- Unusual claim frequencies
- Delayed claim submissions

ReimburseGuard helps auditors and finance teams prioritize inspections by automatically identifying potentially fraudulent claims.

---

## Features

- Fraud Detection using Decision Tree Classification
- Risk Scoring System
- Automated Recommendation Engine
- Data Preprocessing Pipeline
- Feature Importance Analysis
- Confusion Matrix Evaluation
- Anti Data Leakage Implementation
- Class Imbalance Handling using SMOTE

---

## Dataset

The dataset contains 10,000 reimbursement claim records transformed from an insurance claims dataset.

### Features

| Feature | Description |
|----------|-------------|
| jenis_perjalanan | Travel type (Local, Domestic, International) |
| nominal_klaim | Claimed amount |
| nominal_standar | Standard claim amount |
| kelengkapan_dokumen | Document completeness score |
| frekuensi_klaim_bulan | Monthly claim frequency |
| selisih_hari_pengajuan | Submission delay |
| jabatan_pegawai | Employee position |
| riwayat_fraud | Fraud history |

### Target Variable

Fraud Label:

- Fraud (1)
- Normal (0)

A claim is classified as fraud if:

- Claim status = Rejected
- OR claim ratio > 1.8 × standard amount

---

## Machine Learning Pipeline

### 1. Data Preprocessing

- Train-Test Split (80:20)
- Label Encoding
- MinMax Scaling
- SMOTE Oversampling
- Anti Data Leakage Strategy

### 2. Model Training

Algorithm:

- Decision Tree Classifier

Hyperparameters:

```python
DecisionTreeClassifier(
    max_depth=5,
    criterion="gini",
    class_weight="balanced",
    random_state=42
)
```

### 3. Risk Scoring

| Risk Score | Category | Recommendation |
|------------|----------|----------------|
| ≤ 0.30 | Low Risk | Auto Approved |
| 0.31 - 0.60 | Medium Risk | Manual Review |
| > 0.60 | High Risk | Auto Rejected |

---

## Model Performance

| Metric | Result |
|----------|----------|
| Accuracy | 95.65% |
| Precision | 100.00% |
| Recall | 80.36% |
| F1-Score | 89.11% |

---

## Feature Importance

Top features influencing fraud prediction:

| Rank | Feature | Contribution |
|--------|--------|--------|
| 1 | nominal_klaim | 45.21% |
| 2 | nominal_standar | 21.45% |
| 3 | kelengkapan_dokumen | 15.23% |
| 4 | selisih_hari_pengajuan | 9.87% |
| 5 | jenis_perjalanan | 5.64% |

---

## Technology Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn

### Tools

- Jupyter Notebook
- Git
- GitHub

---

## Project Structure

```bash
ReimburseGuard/
│
├── dataset/
│   └── reimbursement_data.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── images/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── risk_score_distribution.png
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── risk_scoring.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/ReimburseGuard.git
```

Move into project directory:

```bash
cd ReimburseGuard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python model_training.py
```

Or open Jupyter Notebook:

```bash
jupyter notebook
```

---

## Results

The model successfully achieved:

- 95.65% Accuracy
- 100% Precision
- 80.36% Recall

The system can support auditors by prioritizing suspicious reimbursement claims and reducing manual verification workload.

---

## Future Improvements

- Random Forest Implementation
- XGBoost Implementation
- SHAP Explainability
- LIME Explainability
- Real-Time Dashboard Integration
- REST API Deployment

---

## Authors

### Group 1

- Andre Muhamad Pajri
- Esa Canoe Alvian Karim
- Jeniva Dewi Milandia

Politeknik Negeri Indramayu

---

## License

This project is developed for educational and research purposes.
