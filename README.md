
# Assignment 2: EEG Epilepsy Detection (BEED Dataset)

## a. Problem Statement

Develop and evaluate machine learning models to detect and classify epileptic seizures using EEG data from the BEED dataset. The goal is to accurately distinguish between healthy and seizure states, as well as different seizure types, based on multichannel EEG recordings.

## b. Dataset Description

The Bangalore EEG Epilepsy Dataset (BEED) contains 16,000 EEG segments (20 seconds each) from 80 adult subjects, evenly split by gender and grouped into four categories: Healthy (0), Generalized Seizures (1), Focal Seizures (2), and Seizure Events (3). Each sample includes 16 EEG channels (X1-X16) and a binary label (y) for seizure presence. Data was recorded at 256 Hz using the standard 10-20 system, providing a balanced and high-quality resource for seizure detection and classification tasks.

## c. Models Used & Comparison Table


Six classical machine learning models were implemented and evaluated using standard metrics. The models include:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

The following table summarizes the evaluation metrics for each model:

| Model                          | Accuracy | Precision | Recall | F1 Score | AUC  | MCC  |
|--------------------------------|----------|-----------|--------|----------|------|------|
| Logistic Regression            | 46.94%   | 0.4986    | 0.4694 | 0.4755   | 0.66 | 0.29 |
| Decision Tree Classifier       | 87.25%   | 0.8746    | 0.8725 | 0.8733   | 0.92 | 0.83 |
| K-Nearest Neighbor Classifier  | 95.94%   | 0.9609    | 0.9594 | 0.9595   | 0.99 | 0.95 |
| Naive Bayes Classifier         | 71.44%   | 0.7173    | 0.7144 | 0.7088   | 0.91 | 0.62 |
| Ensemble Model - Random Forest | 96.31%   | 0.9634    | 0.9631 | 0.9632   | 0.99 | 0.95 |
| Ensemble Model - XGBoost       | 97.12%   | 0.9715    | 0.9712 | 0.9713   | 0.99 | 0.96 |

## d. Model Performance Observations

**Note:** For medical data, Recall (Sensitivity) is the critical metric as it measures the model's ability to identify positive cases. Missing a positive case (false negative) can be dangerous in medical diagnosis.

| Model Name                     | Observation                                                                                 |
|--------------------------------|---------------------------------------------------------------------------------------------|
| Logistic Regression            | **Unacceptable** - Only 46.94% recall; misses nearly half of positive cases; unsafe for medical use |
| Decision Tree Classifier       | **Adequate** - 87.25% recall; acceptable but still misses ~12% of cases; use with caution |
| K-Nearest Neighbor Classifier  | **Excellent** - 95.94% recall; highly reliable at detecting positive cases; recommended for deployment |
| Naive Bayes Classifier         | **Poor** - 71.44% recall; misses ~28% of positive cases; not suitable for medical applications |
| Ensemble Model - Random Forest | **Excellent** - 96.31% recall; superior sensitivity with minimal false negatives; highly recommended |
| Ensemble Model - XGBoost       | **Best** - 97.12% recall; highest sensitivity; catches 97% of positive cases; optimal for medical diagnosis |

---

## How to Run

```bash
# 1. Create a virtual environment (Windows example)
python -m venv .venv

# 2. Activate the environment
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train and save models
python model_train.py

# 5. Launch the Streamlit app
streamlit run streamlit_app.py
```