````markdown
# Quora Insincere Question Classification

NLP project for detecting insincere questions from the Quora Insincere Questions Classification dataset.

The project explores a progression from classical text representations to neural sequence models, with experiments using Bag-of-Words, TF-IDF, pretrained GloVe embeddings, Bidirectional LSTM, and Bidirectional GRU.

## Problem

The task is binary text classification:

- `0` — sincere question
- `1` — insincere question

The original dataset contains more than 1.3 million questions and is highly imbalanced, with insincere questions forming the minority class.

## Project Approach

```text
Raw Questions
      ↓
Text Cleaning
      ↓
Bag-of-Words / TF-IDF
      ↓
Classical ML Baseline
      ↓
Tokenization + Padding
      ↓
Pretrained GloVe Embeddings
      ↓
Bidirectional LSTM
      ↓
Bidirectional GRU
      ↓
Evaluation & Model Comparison
      ↓
FastAPI Inference API
      ↓
Docker Containerization
````

## Experiments

### 1. Bag-of-Words Baseline

Notebook:

```text
notebooks/01_bow_baseline.ipynb
```

A classical NLP baseline using sparse text representations and logistic regression.

### 2. TF-IDF + Neural Network

Notebook:

```text
notebooks/02_tfidf_pytorch.ipynb
```

Uses TF-IDF representations with a neural-network classifier implemented in PyTorch.

### 3. GloVe + BiLSTM + BiGRU

Notebook:

```text
notebooks/03_glove_bilstm_gru.ipynb
```

This notebook implements:

1. Text cleaning
2. Stratified train/validation split
3. Tokenization
4. Integer sequence conversion
5. Padding/truncation
6. Pretrained 100-dimensional GloVe loading
7. Tokenizer-aligned embedding matrix construction
8. Frozen GloVe embedding layer
9. Bidirectional LSTM
10. Bidirectional GRU
11. Precision, recall, F1 and confusion-matrix evaluation
12. Classification-threshold analysis
13. BiLSTM vs BiGRU comparison

A 100,000-question development sample was used to validate the complete pipeline before considering larger-scale training.

## Model Architecture

### BiLSTM

```text
Input Token Sequences
        ↓
GloVe Embedding (100d)
        ↓
Bidirectional LSTM (64 units)
        ↓
Dropout
        ↓
Dense (32, ReLU)
        ↓
Dropout
        ↓
Dense (1, Sigmoid)
```

Current validation result on the 100k development experiment at a 0.5 threshold:

| Model  | Precision | Recall |     F1 |
| ------ | --------: | -----: | -----: |
| BiLSTM |    0.6295 | 0.5331 | 0.5773 |

### BiGRU

```text
Input Token Sequences
        ↓
GloVe Embedding (100d)
        ↓
Bidirectional GRU (64 units)
        ↓
Dropout
        ↓
Dense (32, ReLU)
        ↓
Dropout
        ↓
Dense (1, Sigmoid)
```

Current validation result on the same development experiment at a 0.5 threshold:

| Model | Precision | Recall |     F1 |
| ----- | --------: | -----: | -----: |
| BiGRU |    0.6888 | 0.4818 | 0.5670 |

The BiLSTM currently has the slightly higher F1 score, while the BiGRU has higher precision. The comparison is interpreted using the precision-recall trade-off rather than accuracy alone.

## Why F1, Precision and Recall?

The dataset is strongly class-imbalanced. A model can achieve high overall accuracy while performing poorly on the minority class.

For the current BiLSTM experiment:

```text
Accuracy  : 0.95
Precision : 0.6295
Recall    : 0.5331
F1        : 0.5773
```

Therefore, precision, recall, F1, and the confusion matrix are emphasized when evaluating minority-class performance.

## GloVe

The sequence models use pretrained 100-dimensional GloVe vectors to initialize the embedding layer.

The tokenizer vocabulary is mapped to the corresponding GloVe vectors through an embedding matrix. Words without a matching pretrained vector remain zero-initialized in the current implementation.

The GloVe files are not included in this repository because of their size.

To reproduce the GloVe experiment, download the GloVe 6B vectors, extract `glove.6B.100d.txt`, and place it at:

```text
embeddings/glove.6B.100d.txt
```

The `embeddings/` directory should be excluded from version control.

## Evaluation

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix
* Classification report

The deep-learning experiments also examine different sigmoid decision thresholds to study the precision-recall trade-off.

## Inference API

The trained BiLSTM model was integrated into a standalone inference pipeline and exposed through a FastAPI REST API.

The API accepts a question and returns the predicted class and label.

### API Architecture

```text
Client
   |
   | POST /predict
   v
FastAPI
   |
   v
Inference Pipeline
   |
   +--> Tokenizer
   |
   +--> BiLSTM Model
   |
   v
Prediction
```

### API Endpoint

```text
POST /predict
```

Example request:

```json
{
  "question": "Why do people believe conspiracy theories?"
}
```

Example response:

```json
{
  "prediction": 0,
  "label": "Sincere"
}
```

The API was tested successfully with the trained BiLSTM model.

## Docker Deployment

The BiLSTM inference service was containerized using Docker.

The Docker image contains:

* FastAPI application
* Uvicorn server
* TensorFlow runtime
* Inference pipeline
* Trained model artifacts required for inference

### Build Docker Image

From the project root:

```bash
docker build -t quora-classifier .
```

### Run Docker Container

```bash
docker run --name quora-api -p 8000:8000 quora-classifier
```

The API is then available locally at:

```text
http://localhost:8000
```

### Interactive API Documentation

FastAPI provides interactive Swagger documentation at:

```text
http://localhost:8000/docs
```

The `/predict` endpoint was tested successfully from the running Docker container.

### Deployment Flow

```text
Trained BiLSTM Model
        ↓
inference.py
        ↓
FastAPI
        ↓
Dockerfile
        ↓
Docker Image
        ↓
Docker Container
        ↓
localhost:8000
        ↓
POST /predict
        ↓
Prediction
```

### Deployment Status

* [x] FastAPI inference API
* [x] Dockerfile created
* [x] Docker image built successfully
* [x] Docker container started successfully
* [x] `/docs` endpoint tested
* [x] `/predict` endpoint tested successfully
* [ ] Cloud deployment
* [ ] Kubernetes deployment

## Project Structure

```text
quora-insincere-question-classification/
│
├── app/
│   └── main.py
│
├── data/
│   └── train.csv                  # not committed
│
├── embeddings/
│   └── glove.6B.100d.txt          # not committed
│
├── models/
│   ├── bilstm_best.keras
│   └── gru_best.keras
│
├── notebooks/
│   ├── 01_bow_baseline.ipynb
│   ├── 02_tfidf_pytorch.ipynb
│   └── 03_glove_bilstm_gru.ipynb
│
├── inference.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

Large datasets and pretrained GloVe files are intentionally excluded from version control.

## Key Learning Outcomes

* Real-world imbalanced NLP classification
* Text cleaning and normalization
* Bag-of-Words and TF-IDF representations
* Tokenization and vocabulary construction
* Sequence padding and truncation
* Pretrained word embeddings
* GloVe embedding-matrix construction
* Bidirectional LSTM and GRU architectures
* Binary cross-entropy and sigmoid classification
* Mini-batch neural-network training
* Early stopping and model checkpointing
* Precision-recall trade-offs
* Threshold selection
* Confusion-matrix-based error analysis
* FastAPI-based ML inference
* REST API development
* Docker containerization
* Local container deployment
* Kaggle submission workflows

## Kaggle

The project originated from the Kaggle Quora Insincere Questions Classification competition.

Official competition page:

[https://www.kaggle.com/competitions/quora-insincere-questions-classification](https://www.kaggle.com/competitions/quora-insincere-questions-classification)

## Reproducibility Notes

The repository contains the notebooks and project code used for the experiments. Large datasets and pretrained GloVe files are intentionally excluded from version control.

The deep-learning metrics documented above correspond to the 100,000-question development experiment and should not be interpreted as full-dataset training results.

For the Dockerized inference API, the required runtime dependencies are listed separately in `requirements-docker.txt`.

## Future Work

* Cloud deployment of the FastAPI inference service
* Kubernetes-based model serving
* Multi-model inference using BiLSTM, BiGRU and TF-IDF-based models
* Independent scaling of model-serving components
* Further model and threshold optimization

---

**Project by Simi — MNNIT Allahabad**

````

