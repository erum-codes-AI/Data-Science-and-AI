# Customer Segmentation Using K-Means Clustering

## Project Overview

This project applies **Unsupervised Machine Learning** to perform customer segmentation using the K-Means Clustering algorithm.

The objective is to group customers based on their purchasing behavior and identify different customer segments that can help businesses understand their customers better.

## Technologies & Libraries

- Python
- Pandas – Data manipulation
- NumPy – Numerical operations
- Matplotlib – Data visualization
- Seaborn – Statistical visualization
- Scikit-learn – Machine Learning
- Streamlit – Web application

## Dataset

**Mall Customers Dataset**

The dataset contains customer information such as:
- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

## Machine Learning Algorithm

**K-Means Clustering**

K-Means is an unsupervised learning algorithm that groups similar data points into clusters.

## Project Structure

```text
03-Unsupervised-Learning/
│
├── app.py
├── Files_Lab_4_Customer_Segmentation_(K_means).ipynb
├── Mall_Customers_clean.csv
├── segment_model.pkl
├── requirements.txt
└── README.md
```

## Installation & Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd 03-Unsupervised-Learning
```

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run the application:

```bash
python -m streamlit run app.py
```

## Project Files

- **app.py:** Web application for customer segmentation.
- **Jupyter Notebook:** Machine learning workflow and implementation.
- **Mall_Customers_clean.csv:** Cleaned dataset.
- **segment_model.pkl:** Saved trained clustering model.
- **requirements.txt:** Project dependencies.

## Learning Objectives

- Understanding Unsupervised Learning.
- Exploring and preprocessing customer data.
- Applying K-Means Clustering.
- Understanding customer groups and segmentation.
- Saving and using a trained machine learning model.
- Deploying a machine learning application using Streamlit.

---

**Project Category:** Unsupervised Machine Learning

**Algorithm:** K-Means Clustering
