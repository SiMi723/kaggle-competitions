# 🧪 Jigsaw Toxic Comment Classification

This project is based on the Jigsaw Toxic Comment Classification Challenge. The goal is to build a multi-label classifier that identifies toxic behavior in online comments.

---

## 📌 What I Did

- Explored and visualized class imbalance in toxic labels
- Preprocessed the text (lowercasing, punctuation removal, tokenization)
- Used an Embedding + Bidirectional GRU-based RNN model for multi-label classification
- Trained and validated the model using F1-score as metric
- Prepared submission and tested on Kaggle

---

## 🧠 Model Highlights

- Tokenizer + padded sequences
- Embedding layer using pre-trained GloVe vectors
- BiGRU layers with dropout
- Sigmoid activation for multilabel outputs

---

## 📊 Evaluation

| Method | Public Score | Private Score |
|--------|--------------|---------------|
| Deep Learning (RNN + GRU) | **0.94183** | **0.94440** |

---

## 📂 Structure
jigsaw-toxic-comment-classification/
├── jigsaw_nlp_rnn_live.ipynb
└── README.md

---

## ✅ What I Learned

- How to build multi-label text classifiers
- How to use pre-trained embeddings for better results
- The importance of handling class imbalance in toxic comment datasets
- How to optimize a Kaggle submission

---

📚 *Project by Simi — 2nd Year B.Tech, MNNIT*

