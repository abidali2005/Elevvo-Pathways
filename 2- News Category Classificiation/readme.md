# 📰 AG News Text Classification (NLP + Deep Learning)

This project implements a complete Natural Language Processing (NLP) pipeline to classify news articles into multiple categories using both traditional Machine Learning and Deep Learning approaches.

---

## 📌 Project Objective

Classify news articles into one of the following categories:

- 🌍 World
- 🏀 Sports
- 💼 Business
- 💻 Sci/Tech

The project includes:

- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization
- Logistic Regression baseline model
- Neural Network (Keras)
- Word frequency visualization
- Word clouds per category
- Model comparison

---

## 📂 Dataset

**Dataset Used:** AG News Dataset  

Each record contains:

- `Class Index` → Category label
- `Title` → News headline
- `Description` → News article summary

Example:

| Class Index | Title | Description |
|------------|--------|-------------|
| 3 | Wall St. Bears Claw Back Into the Black | Reuters - Short-sellers... |

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- WordCloud

---

## 🔄 Project Workflow

### 1️⃣ Data Preprocessing



### 2️⃣ Feature Engineering

#### TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
```

#### Keras Tokenization (For Neural Network)

```python
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
```

---

### 3️⃣ Models Implemented

## ✅ Logistic Regression (Baseline)

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
```

**Validation Accuracy:** ~90–91%

---

## ✅ Feedforward Neural Network (Keras)

Architecture:

- Embedding Layer
- GlobalAveragePooling1D
- Dense Layer
- Dropout
- Softmax Output

```python
model = Sequential()
model.add(Embedding(10000, 128, input_length=100))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
```

**Validation Accuracy:** ~87–90%

---

## 📊 Model Comparison

| Model | Validation Accuracy |
|--------|--------------------|
| Logistic Regression | ~90–91% |
| Feedforward NN | ~87–90% |

Observation:
Traditional ML performed competitively compared to simple neural networks on this dataset.

---

## 📈 Data Visualization

### 🔹 Most Frequent Words per Category

Bar plots were generated to visualize top words for each category.

### 🔹 Word Clouds

Word clouds were created to visually represent dominant words per category.

Example insights:

- Sports → team, game, season
- Business → company, market, stock
- World → government, country, war
- Sci/Tech → technology, software, internet

---

## 🧠 Key Learnings

- TF-IDF + Logistic Regression performs strongly for text classification.
- Neural networks require careful tuning to avoid overfitting.
- GlobalAveragePooling reduces model complexity.
- Visualization helps validate preprocessing quality.
- Classical ML can outperform basic deep learning in structured text tasks.

---

## 🚀 Future Improvements

- LSTM / GRU implementation
- Hyperparameter tuning (GridSearch / Keras Tuner)
- Pretrained embeddings (GloVe)
- Transformer-based models (BERT)

---

## 📎 How to Run

1. Install dependencies:
```
pip install pandas numpy nltk scikit-learn tensorflow matplotlib wordcloud
```

2. Run the notebook or script.

---

## 👨‍💻 Author

Abid Ali  
AI / ML Enthusiast  
Focused on NLP, Machine Learning & Deep Learning

---

⭐ If you found this project helpful, feel free to star the repository!
