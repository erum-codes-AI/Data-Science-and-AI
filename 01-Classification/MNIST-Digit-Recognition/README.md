# ✍️ MNIST Handwritten Digit Recognition

A Machine Learning project that recognizes handwritten digits (0–9) using the **MNIST dataset** and a Support Vector Machine (SVM) classifier.

The project includes model training, performance evaluation, image preprocessing, and an interactive Streamlit web application.

<pre>
🚀 LIVE DEMO
</pre>

### 👉 [Click Here to Try the Live Application](https://mnist-digit-recognition-erum.streamlit.app/)

---

A Machine Learning project that recognizes handwritten digits (0–9) using the MNIST dataset and a Support Vector Machine (SVM) classifier.
## Project Overview

The objective is to build a multi-class classification model that can recognize handwritten digits from images.

- **Dataset:** MNIST
- **Total Images:** 70,000
- **Image Size:** 28 × 28 pixels
- **Input Features:** 784 pixels
- **Output Classes:** 10 (digits 0–9)
- **Best Model:** Support Vector Machine (RBF Kernel)
- **Test Accuracy:** 98.33%

## Machine Learning Models

Four classification algorithms were trained and compared.

| Model | Test Accuracy |
|---|---:|
| Support Vector Machine (SVM) | 98.33% |
| Random Forest | 96.69% |
| Logistic Regression | 92.09% |
| Decision Tree | 87.69% |

SVM achieved the highest test accuracy and was selected for the web application.

## Web Application Features

The project includes an interactive Streamlit application.

- Draw handwritten digits using a drawing canvas.
- Upload digit images (PNG, JPG, JPEG).
- Predict digits from 0 to 9.
- Display prediction confidence.
- Show the top three predictions.
- Visualize probabilities for all ten digits.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- SciPy
- Streamlit
- Joblib

## Project Structure

```text
MNIST-Digit-Recognition/
│
├── app.py
├── mnist_svm_model.pkl
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd MNIST-Digit-Recognition
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Model Evaluation

The selected SVM model achieved approximately **98.33% test accuracy** on the MNIST test split.

The model was trained with probability estimation enabled to support confidence displays in the web application.

## Learning Outcomes

- Understanding multi-class classification.
- Working with image datasets.
- Image preprocessing and normalization.
- Training and comparing classification algorithms.
- Evaluating machine learning models.
- Saving and loading trained models with Joblib.
- Building interactive ML applications using Streamlit.

## Author

**Erum Shabir**

AI & Data Science Learner

GitHub: https://github.com/erum-codes-AI