# 📧 Email Spam Classification with Machine Learning & Visual Analytics

This project builds an advanced **Spam Detection System** using Natural Language Processing (NLP) and Machine Learning.  
It not only classifies emails/messages as **Spam** or **Ham (Not Spam)** but also includes **data visualization and model analysis** to better understand how spam detection works.

---

## 📝 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Data Visualizations](#data-visualizations)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)

---

## 🔹 Overview

Spam messages are a major issue in digital communication. This project uses **Machine Learning + NLP techniques** to automatically detect spam messages with high accuracy.

Unlike basic spam classifiers, this project includes:

✔ Text preprocessing  
✔ TF-IDF feature extraction  
✔ Multiple ML models  
✔ Confusion matrix heatmap  
✔ Word frequency visualizations  
✔ Feature importance analysis  


---

## 📂 Dataset

**Dataset Used:** SMS Spam Collection Dataset  
- Total Messages: **5,572**
- Classes:
  - **Ham (0)** — Legitimate messages  
  - **Spam (1)** — Unwanted or promotional messages  

Each row contains:
| Label | Message Text |
|------|--------------|
| 0 | Normal conversation |
| 1 | Promotional or scam message |

Due to dataset licensing, the data file is not included in this repository.  
You can download it from Kaggle:

🔗 https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

After downloading:
1. Extract the ZIP file  
2. Place `spam.csv` in the project folder  
---

## ⚙️ Project Pipeline

1. **Load Dataset**
2. **Visualize Class Distribution**
3. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation & numbers
   - Stopword removal
   - Stemming
4. **Word Frequency Analysis**
5. **TF-IDF Vectorization (with n-grams)**
6. **Train-Test Split**
7. **Train Multiple ML Models**
8. **Evaluate Model Performance**
9. **Visualize Confusion Matrix**
10. **Analyze Important Spam Words**

---

## 📊 Data Visualizations

### 🔹 Spam vs Ham Distribution
Shows whether the dataset is balanced between spam and non-spam messages.

### 🔹 Most Common Words
Displays the top 20 words in:
- Spam messages
- Ham messages

This highlights the language patterns used in spam vs normal communication.

### 🔹 Confusion Matrix Heatmap
A visual representation of:
- Correct predictions
- False positives
- False negatives

### 🔹 Important Words for Spam Detection
Shows which words the model considers strong indicators of spam (based on TF-IDF + Naive Bayes probabilities).

---

## 🤖 Model Training

We trained multiple machine learning models:

| Model | Purpose |
|------|---------|
| Multinomial Naive Bayes | Baseline text classification model |
| Logistic Regression | Linear model for improved generalization |
| Linear SVM | Strong classifier for high-dimensional text data |

Text data was converted into numerical form using:

```python```
TfidfVectorizer(max_features=3000, ngram_range=(1,2))

This allows the model to learn from both single words and word pairs like "free entry".

## 📈 Model Evaluation

The best-performing model achieved:

Metric	Score

Accuracy	97.7%

Precision	100%

Recall	82.7%

F1 Score	90.5%

🔹 Interpretation

High Precision (1.0) → No normal messages were falsely marked as spam

Good Recall (82.7%) → Most spam messages were detected

High Accuracy demonstrates effective spam classification

## 🏆 Results

This project demonstrates:

✔ How NLP preprocessing improves model understanding

✔ How TF-IDF helps convert text into meaningful numeric features

✔ How visualization helps interpret ML model behavior

✔ How different ML models perform on text classification tasks

The system effectively detects spam messages while minimizing false alarms.

## 🚀 Future Improvements

Use deep learning models (LSTM, BERT) for context-based detection

Deploy as a web app for real-time spam filtering

Use larger real-world email datasets (e.g., Enron emails)

Add interactive dashboards using Streamlit or Dash

## ⚡ How to Run

1️⃣ Clone the repository:

git clone <your-repository-link>
cd Email-spam-classification


2️⃣ Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn nltk


3️⃣ Run the script or notebook:

python Email_spam_classification.py

## 🛠 Technologies Used

Python 🐍

Pandas

NumPy

Scikit-learn

NLTK

Matplotlib

Seaborn

## 📌 Author

Nimra Fatima

Machine Learning & Data Science Enthusiast
