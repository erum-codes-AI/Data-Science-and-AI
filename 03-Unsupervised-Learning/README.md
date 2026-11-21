# Customer Segmentation with K-Means

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer00segmentation.streamlit.app/)

**Try the live app:** https://customer00segmentation.streamlit.app/

## Project Overview

This project applies **K-Means clustering**, an unsupervised machine learning algorithm, to segment mall customers based on their annual income and spending score.

The Streamlit application lets users enter a customer's income and spending score, predicts the customer's segment, and displays a suggested marketing approach.

## Live Demo

Open the deployed application:

**[Customer Segmentation Studio](https://customer00segmentation.streamlit.app/)**

## Features

- Predicts a customer segment using a trained K-Means model.
- Interactive sliders for annual income and spending score.
- Visualizes the customer groups in a segmentation map.
- Displays a marketing recommendation for the predicted segment.
- Uses a saved model and scaler without retraining the model when the app starts.

## Customer Segments

The model identifies five customer groups:

| Segment | Description |
|---|---|
| Average customers | Customers with mid-range income and spending scores. |
| VIP customers | Customers with high income and high spending scores. |
| Young big spenders | Customers with lower income but high spending scores. |
| Careful rich | Customers with high income but lower spending scores. |
| Budget savers | Customers with lower income and lower spending scores. |

## Machine Learning Workflow

1. Load and inspect the Mall Customers dataset.
2. Select annual income and spending score as model features.
3. Scale the features using `StandardScaler`.
4. Use the Elbow Method to help choose the number of clusters.
5. Train the K-Means clustering model with five clusters.
6. Analyze the clusters and assign meaningful segment names.
7. Save the trained model, scaler, segment names, and marketing tips.
8. Build and deploy an interactive Streamlit application.

## Dataset

The project uses the **Mall Customers** dataset. The clustering model uses these two features:

- `income` — annual income (in thousands of dollars).
- `spending` — customer spending score.

The dataset also includes customer age, which is used for segment summaries rather than model prediction.

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- K-Means Clustering
- StandardScaler

## Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/erum-codes-AI/Data-Science-and-AI.git
   ```

2. Open the project folder containing `app.py`.

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

Make sure `segment_model.pkl` and `Mall_Customers_clean.csv` are in the same folder as `app.py`.

---

**Developed by Erum Shabir**
