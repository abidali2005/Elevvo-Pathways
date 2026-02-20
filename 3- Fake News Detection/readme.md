# 📰 Fake vs Real News Classification (NLP Project)

## 📌 Project Overview
This project focuses on detecting whether a news article is **Real** or **Fake** using Natural Language Processing (NLP) and Machine Learning techniques.  
The model classifies news articles based on their textual content (title + article body).

The objective is to build a reliable binary classification model and evaluate it using standard performance metrics.

---

## 📂 Dataset
**Source:** Kaggle – Fake and Real News Dataset  

The dataset contains two separate CSV files:
- `True.csv`  → Real news articles
- `Fake.csv`  → Fake news articles

### Dataset Handling:
- Added a `label` column:
  - `1` → Real News
  - `0` → Fake News
- Merged both datasets
- Removed duplicate articles (to prevent data leakage)
- Shuffled dataset before splitting

---

## 🛠️ Data Preprocessing

The following preprocessing steps were applied:

- Combined `title` and `text` columns into a single `content` feature
- Removed duplicate articles
- Removed stopwords
- Text cleaning
- TF-IDF Vectorization

TF-IDF was used to convert text data into numerical feature vectors.

---

## 📊 Exploratory Data Analysis

### Word Cloud Visualization
WordClouds were generated separately for:
- Real News Articles
- Fake News Articles

This helped identify the most frequent and dominant terms in each class, highlighting linguistic differences between real and fake content.

---

## 🤖 Model Training

Two machine learning models were trained:

- Logistic Regression
- Support Vector Machine (Linear SVM)

Train-Test Split:
- 80% Training
- 20% Testing
- Random state fixed for reproducibility

---

## 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- F1-Score
- Classification Report

These metrics provide a balanced evaluation of performance, especially important for binary classification tasks.

---

## 🚀 Results

The model achieved strong classification performance, demonstrating that textual patterns and linguistic features are effective in distinguishing fake and real news.


Example:
- Accuracy: 98%
- F1 Score: 97%

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- WordCloud

---

## 📁 Project Structure

```
├── True.csv
├── Fake.csv
├── Fake_News_Detection.ipynb
├── README.md
```



## 👤 Author

Abid Ali  
Machine Learning & AI Enthusiast  

---

## ⭐ Conclusion

This project demonstrates the application of NLP and Machine Learning techniques to solve a real-world problem — fake news detection.  
By applying proper preprocessing, feature engineering, and model evaluation, the classifier achieves strong predictive performance while maintaining generalization ability.