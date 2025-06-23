# ❓ Quora Insincere Questions Classification

This project is part of a Kaggle competition where the goal is to detect toxic or insincere questions from Quora's question dataset.

🔗 [Official Competition Page](https://www.kaggle.com/competitions/quora-insincere-questions-classification)

---

## 📌 What I Did

- Explored the dataset using pandas
- Preprocessed text data (tokenization, padding, etc.)
- Used both:
  - **Classical NLP methods** (Bag-of-Words + Logistic Regression)
  - **Deep Learning models** (Embedding + RNN)
- Compared model performance
- Generated submission in `submission.csv`

---

## 🧪 Dataset

| File | Description |
|------|-------------|
| `train.csv` | Training data with questions and labels |
| `test.csv` | Test questions without labels |
| `sample_submission.csv` | Submission format required by Kaggle |

---

## 🧠 Models Used

### Classical Approach
- Bag-of-Words (CountVectorizer / TF-IDF)
- Logistic Regression
- Naive Bayes

### Deep Learning
- Keras Embedding Layer (GloVe or learned)
- Bidirectional LSTM / GRU
- Dropout and Dense layers
- Sigmoid output for binary classification

---

## 📊 Evaluation

| Notebook | Model Type        | Public Score | Private Score |
|----------|-------------------|--------------|---------------|
| `quora_text_classification_live.ipynb` | Classical ML (BoW + LR) | 0.39729 | **0.39729** |
| `quora_classification_deep_learning_live.ipynb` | Deep Learning (RNN) | 0.40849 | **0.41828** |

> Scores are from the Kaggle leaderboard and reflect individual model submissions.

---

## 📂 Project Structure

---

## ✅ What I Learned

- How to preprocess and clean real-world NLP datasets
- How to build text classification pipelines using both classical and deep learning methods
- How to prepare submission files for Kaggle
- How to debug low leaderboard scores (e.g., checking submission format, label alignment)
- How to push a project to GitHub for portfolio building

---

📚 *Project by Simi — 2nd Year B.Tech, MNNIT*

