# House Price Prediction in Sukkur — Regression Project

## Project Overview
This project uses Machine Learning regression algorithms to predict house prices in Sukkur, Pakistan.

The model uses property details such as location, area, bedrooms, bathrooms, house age, condition, and other features to predict the house price in lakh rupees.

**Problem Type:** Supervised Learning — Regression  
**Target Variable:** `price_lakh`

> This is a practice dataset created for an AI and Data Science course. It is designed for learning and is not a verified source of real-world property prices.

## Dataset
- File: `sukkur_house_prices.csv`
- Original records: 1,215
- Target: House price in lakh rupees
- Features include location, area, bedrooms, bathrooms, floors, age, condition, garage, and more.

## Technologies and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Project Workflow

1. Define the regression problem.
2. Load and inspect the dataset.
3. Clean duplicate rows, inconsistent locations, invalid values, and outliers.
4. Perform Exploratory Data Analysis (EDA).
5. Prepare numerical, nominal, and ordinal features.
6. Handle missing values, scaling, and encoding using pipelines.
7. Split data into training and testing sets.
8. Train and compare regression models using 5-fold cross-validation.
9. Tune the Gradient Boosting model using GridSearchCV.
10. Save the final model and predict prices for new houses.

## Models Compared
- Baseline (Dummy Regressor)
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Final Model Results

**Selected Model:** Tuned Gradient Boosting Regressor

| Metric | Test Result |
|---|---:|
| MAE | 8.74 lakh Rs |
| RMSE | 13.05 lakh Rs |
| R² Score | 0.960 |

The model achieved an R² score of 0.960 on the test set. These results apply to this practice dataset and should not be interpreted as real-world price prediction accuracy.

## Project Files

| File | Description |
|---|---|
| `Houseing Price.ipynb` | Complete Jupyter Notebook |
| `sukkur_house_prices.csv` | Dataset |
| `app.py` | Streamlit web application |
| `house_price_model.joblib` | Saved trained preprocessing and prediction pipeline |
| `README.md` | Project documentation |
## How to Run

1. Clone or download this repository.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Keep the notebook and CSV dataset in the same folder.
4. Open the notebook in Jupyter Notebook or Google Colab.
5. Run the cells from top to bottom.


## Streamlit Web Application

This project also includes a Streamlit application that allows users to predict house prices by entering property details.

### Features
- Select a Sukkur location.
- Enter area, bedrooms, bathrooms, floors, and house age.
- Select property condition and additional features.
- Get an estimated house price in lakh rupees.

### Run the Application

1. Install the required libraries:

   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```

2. Make sure `app.py` and `house_price_model.joblib` are in the same folder.

3. Open the terminal in the project folder.

4. Run:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

**Note:** Predictions are educational estimates based on a course practice dataset, not verified market valuations.

## Learning Outcomes
- Understanding regression problems.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature encoding and scaling.
- Scikit-learn Pipelines and ColumnTransformer.
- Train/test split and cross-validation.
- Comparing regression algorithms.
- Hyperparameter tuning with GridSearchCV.
- Model evaluation using MAE, RMSE, and R².
- Saving and loading ML models using Joblib.

---

**Project Category:** Machine Learning — Regression  
**Purpose:** Learning and portfolio development.