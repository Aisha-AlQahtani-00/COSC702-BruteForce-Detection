# COSC702-BruteForce-Detection
# Benchmarking ML algorithms for brute force attack detection on Azure authentication logs
**Course:** COSC 702: Advanced AI-Driven Software Engineering 
**Institution:** Khalifa University 
**Submitted to:** Dr. Jamal Bentahar
**Authors:** Aisha AlQahtani, Salwa Mohammed Razaulla  

---

## Overview 

This project presents a comprehensive comparative analysis of ten Machine Learning algorithms, 5 supervised and 5 unsupervised, for detecting brute-force attacks in Microsoft Azure logs. The frameowrk includes a feature-level ablation study, contatmintaion rate sensitivet analysis, and scenario-based robustness evaluation across six data conditions. 

---



## Repository Structure

```
COSC702-BruteForce-Detection/
│
├── config.py               #Shared imports, constants and helper functions.
├── Main_AD_Code.py         #Main comparative analysis, trains all 10 models.
├── Ablation_Study.py       #Ablation study, 3 experiments. 
├── Scenarios_Test.py       #Scenario-based testing across 6 datasets. 
│
├── datasets/
│   ├── Training/           #13k labeled training dataset.
│   └── Testing/            #Scenario testing datasets (1k to 100k).
│
└── outputs/                #Generated charts. 
```
---

# Run Order 

Ensure all files are in the same directory before running. 'confing.py' is imported automatically by all other files, do not run it directly. 

1. Main_AD_Code.py  -> trains all 10 midels, saves .pkl files.
2. Ablation_Study.py -> feature importance and sensitivty experiments.
3. Scenarios_Test.py -> scenario_based robustness testing and ground truth evaluation.

---

## Dependencies 

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib
```

---

## Dataset Description 

| Dataset | Split | Records | Purpose |
|---|---|---|---|
| Dataset13k | Training | 13,010 | Model training & evaluation |
| Dataset1k | Testing | 1,006 | Small scale scenario test |
| Dataset15k | Testing | 15,000 | Medium scale scenario test |
| Dataset20k | Testing | 20,611 | Large scale scenario test |
| Dataset100k | Testing | 101,104 | Very large scale scenario test |
| Healthy Logs | Testing | 1,000 | False positive evaluation |
| Empty Dataset | Testing | 0 | Edge case robustness test |

---

## Key Findings 

- Supervised models significantly outperformed unsupervised approaches, Gradient Boosting achived F1=0.9963 vs Isolation Forest's F1=0.9779.
- Geographic location proved a stronger brute-force indicator than login failure status, challenging conventional assumptions.
- Ellipltic Envelope and Local Outlier Factors are incompatible with sparse TF-IDF feature spaces.
- Supervised models demonstrated near-perfect calibration on unseen data at 100k scale.

---

## Feature Columns Used

| Feature | Cybersecurity Role |
|---|---|
| Status | Login outcome — Success, Failure, Interrupted |
| Location | Geographic anomaly detection |
| IP Address | Network-level anomaly detection |
| Application | Target surface identification |

---

## Prior Work 

This study extends the following published works made by the authors: 

- AlQahtani, A. and Taher, F. (2023). *AI implementations in cloud-based sign-in logs to detect brute force attack attempts.* ICDSIS 2023. DOI: 10.1049/icp.2024.0494
- AlQahtani, A. and Taher, F. (2025). *Detecting User Sign-in Anomalies in Cloud-based Logs Using Machine Learning Techniques.* CCNCPS 2025. DOI: 10.1109/CCNCPS66785.2025.11135742
