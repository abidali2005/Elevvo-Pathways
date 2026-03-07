# Elevvo Pathways Internship — Natural Language Processing Projects

## Overview

This repository contains the projects completed during the **Elevvo Pathways Internship** focused on **Natural Language Processing (NLP)**.
The goal of this internship was to apply modern NLP techniques to real-world datasets using Python and widely used machine learning libraries.

Throughout the internship, multiple NLP tasks were implemented including **text preprocessing, classification, named entity recognition, topic modeling, and transformer-based NLP**. Each project demonstrates different NLP concepts such as **binary classification, multiclass classification, sequence labeling, unsupervised topic modeling, and abstractive summarization**.

These projects showcase practical experience with **data preprocessing, feature engineering, machine learning models, evaluation metrics, and visualization techniques**.

---

## Repository Structure

```
Elevvo-Pathways/
│
├── 1-Text-Preprocessing
│   ├── Text_Preprocessing.ipynb
│   └── README.md
│
├── 2-News-Category-Classification
│   ├── News_Category.ipynb
│   ├── train.csv
│   ├── test.csv
│   └── README.md
│
├── 3-Fake-News-Detection
│   ├── Fake_News_Detection.ipynb
│   ├── Fake.csv
│   ├── True.csv
│   └── README.md
│
├── 4-Named-Entity-Recognition
│   ├── Named_Entity_Recognition.ipynb
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── README.md
│
├── 5-Topic-Modeling
│   ├── Topic_Modeling.ipynb
│   └── README.md
│
├── 6-Question-Answering-With-Transformers
│   ├── QA_Transformers.ipynb
│   └── README.md
│
└── 7-Text-Summarization
    ├── News_Summarization_T5.ipynb
    └── README.md
```

---

# Task 1: Sentiment Analysis on Product Reviews

## Description

Sentiment analysis is a binary classification task used to determine whether a piece of text expresses a **positive or negative sentiment**. In this project, product reviews are analyzed and classified into sentiment categories.

The main objective is to build a machine learning model capable of understanding customer opinions from textual reviews.

---

## Dataset

Product Reviews Dataset (Kaggle)

The dataset contains:

```
review_text
sentiment
```

Where:

* **review_text** contains customer review text
* **sentiment** indicates whether the review is positive or negative

---

## Methodology

### Text Preprocessing

The following preprocessing steps were applied:

```
Tokenization
Lowercasing
Stopword removal
Lemmatization
```

These steps help convert raw text into a cleaner format suitable for machine learning models.

---

### Feature Extraction

Text data was converted into numerical vectors using:

```
TF-IDF (Term Frequency – Inverse Document Frequency)
```

TF-IDF helps represent important words while reducing the impact of very common words.

---

### Model Training

A classification model was trained to predict sentiment using:

```
Logistic Regression
```

Optionally, a **simple feedforward neural network using Keras** was explored to compare performance with traditional models.

---

### Evaluation Metrics

Model performance was evaluated using:

```
Accuracy
Precision
Recall
F1 Score
```

---

### Bonus Visualization

Word frequency visualizations were created to better understand the dataset:

```
Bar plots
Word clouds
```

These visualizations highlight the most common words in positive and negative reviews.

---

# Task 2: News Category Classification

## Description

This project focuses on **multiclass text classification** where news articles are categorized into predefined topics.

Possible categories include:

```
Sports
Business
Politics
Technology
Entertainment
```

---

## Dataset

AG News Dataset (Kaggle)

The dataset includes:

```
title
description
category
```

---

## Preprocessing Steps

```
Tokenization
Lowercasing
Stopword removal
Lemmatization
```

The goal of preprocessing is to clean and normalize textual data before feature extraction.

---

## Feature Engineering

Two main approaches were explored:

```
TF-IDF Vectorization
Word Embeddings
```

TF-IDF converts text into sparse vectors based on word importance.

---

## Model Training

Multiple classification models were trained:

```
Logistic Regression
Support Vector Machine (SVM)
Random Forest
```

Optionally:

```
XGBoost
LightGBM
```

---

## Evaluation Metrics

Performance was evaluated using:

```
Accuracy
Confusion Matrix
Classification Report
```

---

# Task 3: Fake News Detection

## Description

Fake news detection aims to classify news articles as **real or fake** based on textual content.

This task demonstrates how NLP techniques can be applied to identify misinformation.

---

## Dataset

Fake and Real News Dataset (Kaggle)

The dataset contains:

```
title
text
label
```

Where the label indicates whether the article is **real or fake**.

---

## Text Preprocessing

```
Remove stopwords
Lemmatization
Tokenization
Text vectorization
```

---

## Feature Extraction

Text was converted to numerical vectors using:

```
TF-IDF
```

---

## Model Training

Binary classification models were trained:

```
Logistic Regression
Support Vector Machine (SVM)
```

---

## Model Evaluation

Performance was evaluated using:

```
Accuracy
Precision
Recall
F1 Score
```

---

## Bonus Visualization

To understand patterns in fake vs real news:

```
Word clouds
```

These visualizations highlight commonly used words in both categories.

---

# Task 4: Named Entity Recognition (NER)

## Description

Named Entity Recognition is a **sequence labeling task** used to identify entities such as:

```
Persons
Organizations
Locations
Dates
```

within text.

---

## Dataset

CoNLL-2003 Dataset

The dataset contains labeled tokens with entity tags.

Example:

```
EU B-ORG
rejects O
German B-MISC
call O
```

---

## Approach

Two approaches were explored:

```
Rule-based NER
Model-based NER using spaCy
```

---

## Implementation

Entities were extracted and categorized using spaCy's pretrained models.

Extracted entities include:

```
PERSON
ORG
GPE
DATE
```

---

## Bonus

Entity visualization was performed using:

```
spaCy displacy
```

This tool visually highlights named entities directly in text.

---

# Task 5: Topic Modeling on News Articles

## Description

Topic modeling is an **unsupervised learning technique** used to discover hidden themes in a collection of documents.

This project identifies dominant topics within news articles.

---

## Dataset

BBC News Dataset

---

## Preprocessing Steps

```
Tokenization
Lowercasing
Stopword removal
```

---

## Topic Modeling Algorithm

The primary algorithm used was:

```
Latent Dirichlet Allocation (LDA)
```

LDA identifies groups of words that frequently occur together and treats them as topics.

---

## Output

For each topic, the model generates:

```
Top keywords representing that topic
```

---

## Bonus

Visualization techniques include:

```
Word clouds
pyLDAvis interactive visualization
```

Additionally, results were compared with:

```
Non-negative Matrix Factorization (NMF)
```

---

# Task 6: Question Answering with Transformers

## Description

This project implements a **transformer-based question answering system** capable of extracting answers from a passage.

The system receives:

```
Context passage
Question
```

and returns the **answer span** from the context.

---

## Dataset

SQuAD v1.1 Dataset

---

## Model

Pretrained transformer models were used:

```
BERT
DistilBERT
```

via the **Hugging Face Transformers library**.

---

## Workflow

```
Context + Question
        ↓
Tokenization
        ↓
Transformer Model
        ↓
Extracted Answer Span
```

---

## Evaluation Metrics

Model performance was measured using:

```
Exact Match (EM)
F1 Score
```

---

# Task 7: Text Summarization

## Description

Text summarization automatically generates concise summaries from long documents.

This project implements **abstractive summarization** using transformer models.

---

## Dataset

CNN-DailyMail News Dataset

The dataset includes:

```
article
highlights
```

Where:

* **article** is the full news article
* **highlights** is the human-written summary

---

## Model

The summarization system uses:

```
T5 Transformer Model
```

which follows an **encoder–decoder architecture**.

---

## Preprocessing

Minimal preprocessing was applied to preserve natural language:

```
Remove newline characters
Remove extra spaces
Add task prefix: summarize:
```

---

## Tokenization and Truncation

Input length was limited to ensure efficient processing:

```
Maximum input length: 256 tokens
Maximum summary length: 64 tokens
```

---

## Evaluation

Summary quality was measured using:

```
ROUGE-1
ROUGE-2
ROUGE-L
```

These metrics measure overlap between generated summaries and reference summaries.

---

# Technologies Used

The following tools and libraries were used throughout the internship:

```
Python
Pandas
NumPy
NLTK
spaCy
Scikit-learn
TensorFlow / Keras
Gensim
pyLDAvis
Hugging Face Transformers
Matplotlib
Seaborn
```

---

# Key Skills Demonstrated

This repository demonstrates proficiency in:

```
Natural Language Processing
Text preprocessing and cleaning
Machine learning for text classification
Binary and multiclass classification
Named Entity Recognition
Topic modeling
Transformer-based NLP
Model evaluation and visualization
```

---

# Conclusion

This repository represents the completion of multiple NLP tasks during the **Elevvo Pathways Internship**. Each project demonstrates a different aspect of natural language processing, ranging from traditional machine learning techniques to modern transformer-based approaches.

Through these tasks, practical experience was gained in **data preprocessing, feature extraction, model training, evaluation metrics, and visualization techniques**. These projects collectively highlight the ability to apply NLP methods to real-world datasets and build end-to-end text analysis systems.

---

Author: Abid Ali
