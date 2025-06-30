igsaw Toxic Comment Classification Challenge 🤔🔒

This project is based on Kaggle's "Toxic Comment Classification Challenge," where the task is to identify and classify different types of toxicity in online comments.

🎡 Objective

To build a multi-label classifier that can identify toxic comments belonging to the following categories:

Toxic

Severe Toxic

Obscene

Threat

Insult

Identity Hate

💡 What I Did

Explored and preprocessed text data

Applied basic NLP cleaning (lowercasing, punctuation removal, stopword filtering)

Tokenized text and padded sequences for model input

Built a Recurrent Neural Network (RNN) using Keras

Used sigmoid activation for multi-label classification

📊 Results

Notebook

Method

Public Score

jigsaw_nlp_rnn_cleaned.ipynb

RNN (GRU)

~0.9828 (F1)

Note: This was a learning-focused implementation. Still exploring advanced techniques like attention and BERT.

📁 Files

jigsaw_nlp_rnn_cleaned.ipynb – Core notebook with preprocessing, model building, and predictions

📖 Dataset

Jigsaw Toxic Comment Classification Challenge

✨ Learnings

Multi-label classification with RNNs

Better understanding of sigmoid vs softmax

Dataset imbalance handling strategies

💪 Next Steps

Add validation strategies (StratifiedKFold, F1 optimization)

Try Bidirectional GRU + GloVe embeddings

Explore Transformer-based models like BERT

Project by Simi – Feel free to connect or suggest improvements!
