# Named Entity Recognition (NER) using CoNLL-2003 Dataset

## 📌 Project Overview

This project implements a Named Entity Recognition (NER) system using the CoNLL-2003 dataset.  

The goal is to automatically identify and classify named entities in text into the following categories:

- PERSON (PER)
- LOCATION (LOC)
- ORGANIZATION (ORG)
- MISCELLANEOUS (MISC)

Two approaches were implemented:

1. Rule-Based NER
2. Model-Based NER using Conditional Random Fields (CRF)

Entity visualization is performed using spaCy.

---

## 📂 Dataset

Dataset Used: CoNLL-2003 (English NER Dataset)

The dataset contains three files:

- train.txt
- valid.txt
- test.txt

Each file contains tokens along with their corresponding NER tags in IOB format:

Example:

Barack  B-PER  
Obama   I-PER  
visited O  
France  B-LOC  

Tag Meaning:
- B-XXX → Beginning of entity
- I-XXX → Inside entity
- O → Outside entity

---

## ⚙️ Technologies Used

- Python
- pandas
- sklearn-crfsuite
- seqeval
- spaCy

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

- Parsed CoNLL formatted text files
- Grouped tokens into sentence-level structures
- Extracted word-label pairs

---

### 2️⃣ Rule-Based Approach

A simple rule-based NER system was implemented using:

- Capitalization rules
- Basic heuristic logic

This approach does not learn from data and serves as a baseline model.

---

### 3️⃣ Feature Engineering

For the CRF model, the following features were extracted for each word:

- Lowercase word
- Is uppercase
- Is title case
- Is digit
- Previous word features
- Next word features
- Beginning/End of sentence indicators

---

### 4️⃣ Model-Based Approach (CRF)

A Conditional Random Field (CRF) model was trained using:

- Algorithm: LBFGS
- Maximum iterations: 100

CRF was chosen because it performs well for sequence labeling tasks such as NER.

---

### 5️⃣ Model Evaluation

Evaluation metrics used:

- Precision
- Recall
- F1-score

Entity-level evaluation was performed using seqeval.

The CRF model significantly outperformed the rule-based system.

---

### 6️⃣ Entity Visualization

Entities predicted by the CRF model were visualized using spaCy’s displacy module.

This provides color-coded highlighting of extracted entities directly inside the notebook.

---

## 📊 Results Summary

| Approach      | Performance |
|--------------|------------|
| Rule-Based   | Low Accuracy |
| CRF Model    | High Accuracy |

The CRF model captures contextual dependencies between words, leading to improved performance compared to rule-based heuristics.

---

## 📁 Project Structure

NER_Project/
│
├── train.txt
├── valid.txt
├── test.txt
├── ner_project.ipynb
└── README.md

---

## 🚀 How to Run

1. Install required libraries:

pip install pandas sklearn-crfsuite seqeval spacy

2. Open ner_project.ipynb
3. Run all cells
4. View evaluation results and entity visualization

---

## 🎯 Key Learnings

- Understanding of sequence labeling problems
- Implementation of rule-based vs machine learning NER
- Feature engineering for CRF models
- Entity-level evaluation metrics
- Visualization of named entities using spaCy

---

## 📌 Conclusion

This project demonstrates the implementation of Named Entity Recognition using both heuristic and machine learning approaches.  

The CRF-based model effectively captures contextual patterns and significantly improves entity extraction accuracy over rule-based methods.

Future improvements may include:
- BiLSTM or Transformer-based models (e.g., BERT)
- Hyperparameter tuning
- Deployment as a web application

---

## 👤 Author

Abid Ali  
Machine Learning & NLP Enthusiast  
Faisalabad, Pakistan