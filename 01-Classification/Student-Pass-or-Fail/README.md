# Student Pass or Fail Prediction | Machine Learning Classification

## Project Overview

This project uses Machine Learning classification to predict whether a student is likely to **Pass or Fail** based on academic performance and other student-related information.

The objective is to identify students who may be at risk of failing so that trainers can provide additional support.

This project was completed as part of my Data Science and AI coursework.

## Problem Statement

This is a **Binary Classification** problem.

- **Input (Features):** Study hours, attendance, quiz average, assignments, education, laptop availability, internet access, and other student information.
- **Target:** `result`
- **Classes:** Pass (1) and Fail (0)

## Dataset

- Dataset: `saylani_student_results.csv`
- Records after cleaning: 1,000
- Original dataset includes missing values, duplicate records, inconsistent text, and invalid values.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Project Workflow

### 1. Data Loading and Inspection
- Loaded the dataset using Pandas.
- Inspected dataset shape, columns, data types, and statistical summaries.

### 2. Data Cleaning
- Removed duplicate records.
- Standardized Pass/Fail labels and city names.
- Handled invalid attendance and study-hour values.
- Removed unnecessary columns (`student_id` and `final_exam_score`).
- Identified missing values.

### 3. Exploratory Data Analysis (EDA)
- Pass vs Fail distribution.
- Attendance comparison using boxplots.
- Study hours vs quiz marks using scatterplots.
- Pass rate comparison based on laptop availability.
- Group-based analysis using Pandas.

### 4. Data Preprocessing
- Converted Pass/Fail into binary values (1/0).
- Encoded Yes/No categorical features.
- Applied ordinal encoding to education levels.
- Applied one-hot encoding to city.
- Filled missing numerical values using median.
- Filled missing laptop values using mode.

### 5. Train/Test Split
- Training data: 80%
- Testing data: 20%
- Used `stratify=y` to maintain the Pass/Fail class distribution.
- Used `random_state=42` for reproducibility.

### 6. Model Training

Three classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### 7. Model Evaluation

The models were evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

Special attention was given to **Fail-class recall**, because identifying students at risk of failing is an important objective.

### 8. Threshold Adjustment

Experimented with changing the classification threshold from 0.50 to 0.65.

This increased the recall for failing students from approximately 74% to 83%, while introducing more false alarms.

### 9. Model Comparison

| Model | Accuracy | Fail Recall | Fail F1 |
|---|---:|---:|---:|
| Logistic Regression | 83.5% | 73.9% | 75.6% |
| Decision Tree | 82.0% | 73.9% | 73.9% |
| Random Forest | 83.0% | 66.7% | 73.0% |

**Selected Model: Logistic Regression**

It achieved the highest Fail-class F1-score among the three tested models.

### 10. Model Saving and Prediction
- Saved the selected model using Joblib.
- Saved the training feature columns.
- Created a custom prediction function.
- Tested predictions using manually entered student information.

### 11. Streamlit Application

Developed a simple interactive web application using Streamlit.

Users can enter student details and receive:
- Predicted Pass/Fail status.
- Estimated probability of passing.
- A visual probability indicator.

## Project Files

| File | Description |
|---|---|
| `Lab 2 - Student Pass or Fail (Beginner).ipynb` | Complete Jupyter Notebook |
| `saylani_student_results.csv` | Dataset |
| `student_app.py` | Streamlit application |
| `student_model.pkl` | Saved classification model |
| `student_model_columns.pkl` | Model feature columns |
| `README.md` | Project documentation |

## How to Run the Project

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit jupyter
```

### 2. Run the Notebook

Open the `.ipynb` file in Google Colab or Jupyter Notebook.

Keep the CSV dataset in the same directory.

### 3. Run the Streamlit Application

Make sure the model files are in the same directory as `student_app.py`.

```bash
streamlit run student_app.py
```

The application will open in your browser.

## Learning Outcomes

Through this project, I practised:

- Data cleaning and preprocessing.
- Exploratory Data Analysis and visualization.
- Categorical encoding and missing-value handling.
- Training multiple classification models.
- Understanding confusion matrices and classification metrics.
- Precision-recall trade-offs and threshold adjustment.
- Saving and loading trained ML models.
- Building a simple ML web application using Streamlit.

## Note

This is an educational practice project using a course-provided dataset. The model predictions are experimental and should not be treated as definitive judgments about students.
