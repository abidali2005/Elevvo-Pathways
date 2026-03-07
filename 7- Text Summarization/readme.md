# CNN-DailyMail News Summarization using T5 (NLP Project)

## Project Overview

This project implements **Abstractive Text Summarization** using the **T5 Transformer model**. The system takes long news articles from the CNN-DailyMail dataset and generates concise summaries automatically using a pretrained encoder–decoder architecture.

The objective is to train a Natural Language Processing model capable of understanding long documents and producing meaningful summaries while preserving key information.

---

## Dataset

Dataset used: **CNN-DailyMail News Dataset**

The dataset contains three columns:

* **id** – Unique identifier for each news article
* **article** – Full news article text (input)
* **highlights** – Human-written summary (target output)

For model training, only the following columns are required:

```
article
highlights
```

The **id column is removed** because it does not contribute to model learning.

---

## Project Objective

The goal of this project is to build a summarization model that can:

* Read long news articles
* Understand the context of the text
* Generate short summaries automatically
* Evaluate summary quality using ROUGE metrics

---

## Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* HuggingFace Datasets
* ROUGE Score Library
* Pandas

---

## Model Architecture

The project uses the **T5-small transformer model**, which follows an **encoder–decoder architecture**.

### Encoder

Reads and understands the input article.

### Decoder

Generates the summary based on encoded information.

T5 models require a task prefix. Therefore, the input article is formatted as:

```
summarize: <article text>
```

---

## Data Preprocessing

Minimal preprocessing is applied because transformer models are already trained on natural language.

Steps performed:

1. Remove unnecessary columns (drop `id`)
2. Remove empty or null rows
3. Remove newline characters
4. Remove extra spaces
5. Add the task prefix for T5:

```
summarize: article_text
```

Heavy preprocessing such as stopword removal, stemming, or lowercasing is **not performed** because it can damage sentence structure.

---

## Tokenization and Truncation

Transformer models accept text in the form of tokens.

The tokenizer converts text into token IDs before feeding it into the model.

To handle long documents, the input is truncated to model limits.

```
Maximum input length: 256 tokens
Maximum summary length: 64 tokens
```

Truncation ensures the model processes text within memory constraints.

---

## Training Process

The training process involves the following steps:

1. Load dataset
2. Clean and preprocess text
3. Convert dataset to HuggingFace format
4. Tokenize input articles and summaries
5. Split dataset into training and evaluation sets
6. Train the T5 model using the HuggingFace Trainer API

Because the project runs on **CPU**, the following optimizations are applied:

* Batch size = 1
* Reduced sequence length
* Single training epoch

These changes ensure the model can train without requiring GPU hardware.

---

## Model Inference

After training, the model can generate summaries for new articles using the following pipeline:

```
Article → Tokenization → Encoder → Decoder → Generated Summary
```

Beam search is used during generation to improve summary quality.

---

## Evaluation

The generated summaries are evaluated using **ROUGE metrics**, which compare model output with human-written summaries.

Metrics used:

* **ROUGE-1** – Measures word overlap
* **ROUGE-2** – Measures phrase overlap
* **ROUGE-L** – Measures longest common subsequence similarity

Higher ROUGE scores indicate better summarization performance.

---

## Project Structure

```
cnn-news-summarization/
│
├── dataset/
│   └── cnn_dailymail.csv
│
├── summarization.py
│
├── outputs/
│   └── trained_model
│
└── README.md
```

---

## Installation

Install the required libraries using pip:

```
pip install transformers
pip install datasets
pip install rouge_score
pip install torch
pip install pandas
```

---

## Running the Project

Run the main script:

```
python summarization.py
```

The script will:

1. Load the dataset
2. Preprocess the data
3. Train the T5 model
4. Generate summaries
5. Evaluate results using ROUGE scores

---

## Example

Input Article:

```
The Pakistan cricket team secured a dramatic victory after chasing a challenging target in the final overs of the match.
```

Generated Summary:

```
Pakistan wins match after successful run chase.
```

---

## Limitations

* Training on CPU can be slow.
* The dataset is large, which may require reducing sample size for faster experimentation.
* Larger models such as BART or PEGASUS may produce better summaries but require GPU resources.

---


## Conclusion

This project demonstrates how transformer-based models can be used for abstractive text summarization. By leveraging pretrained architectures such as T5 and evaluating performance using ROUGE metrics, it is possible to generate meaningful summaries from long news articles efficiently.

The project provides a practical implementation of modern NLP techniques for document summarization tasks.
