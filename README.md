# 📰 Fake News Detection – NLP Project

## 📌 Overview

This project focuses on building a machine learning pipeline to classify news headlines as **real (1)** or **fake (0)**.

We explored multiple NLP approaches:

* Classical models (TF-IDF + ML)
* Feature engineering (combined representations)
* Deep learning (Embedding + BiLSTM)
* Transformer models (BERT)

---

## 📂 Project Structure

* `model1.ipynb` → Baseline models (BoW, TF-IDF, multiple algorithms)
* `model2.ipynb` → Improved preprocessing + feature engineering
* `model3.ipynb` → BiLSTM embedding model
* `model4.ipynb` → Transformer-based model (DistilBERT)

---

## ⚙️ Data

Dataset columns:

* `label`: 0 = fake, 1 = real
* `title`: news headline

We used a consistent **80/20 train-validation split**:

```python
train_test_split(test_size=0.2, random_state=42)
```

---

## 🧪 Models & Experiments

### 1. TF-IDF + Linear SVM (Best Classical Model)

We created a **combined feature representation**:

```python
X_train_combined = hstack([X_train_TF_raw, X_train_TF_S])
X_val_combined = hstack([X_val_TF_raw, X_val_TF_S])
```

#### Hyperparameters:

```python
TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    min_df=3,
    max_df=0.85
)
```

#### ✅ Result:

* **Accuracy: 0.9537**

#### 💡 Insight:

Combining raw + stemmed TF-IDF improves performance by capturing both:

* semantic meaning
* normalized word forms

---

### 2. Embedding Model (BiLSTM)

We implemented:

* Tokenization
* Embedding layer
* Bidirectional LSTM

#### ✅ Result:

* **Accuracy: 0.9687**

#### 💡 Insight:

Embedding models capture:

* word order
* contextual relationships
  which improves performance over TF-IDF.

---

### 3. Transformer Model (DistilBERT) ⭐

We used:

* `distilbert-base-uncased`
* Minimal preprocessing
* Pretrained contextual embeddings

#### ✅ Final Results:

* **Accuracy: 0.9848**

#### 📊 Classification Report:

* Precision: ~0.98–0.99
* Recall: ~0.98–0.99
* F1-score: ~0.98

#### 🔍 Confusion Matrix:

```
[[3468   47]
 [  57 3259]]
```

#### 💡 Insight:

BERT significantly outperforms all other models because:

* it understands **context**
* it captures **word meaning in sentences**
* it handles **negation and subtle language patterns**

---

## 📊 Model Comparison

| Model                   | Accuracy      |
| ----------------------- | ------------- |
| TF-IDF + SVM (combined) | 0.9537        |
| Embedding (BiLSTM)      | 0.9687        |
| **BERT (DistilBERT)**   | **0.9848** 🔥 |

---

## 🧠 Key Learnings

* TF-IDF is strong but ignores context
* Feature engineering improves classical models
* Embeddings capture semantic relationships
* Transformers provide **state-of-the-art performance**

---

## 🎯 Final Conclusion

* Classical ML plateaued around **95% accuracy**
* Deep learning improved performance to **~97%**
* Transformer models achieved **~98.5% accuracy**

👉 **Best model: DistilBERT (98.48%)**

---

## 🚀 Future Improvements

* Fine-tune BERT with more epochs
* Use full article text instead of only titles
* Ensemble multiple models
* Hyperparameter optimization

---

## 👥 Team Strategy

* Built a shared baseline model
* Split advanced experiments across team members
* Compared multiple approaches
* Selected best-performing model for final submission

---

## 📎 Final Output

The final model is used to generate predictions for:

```
validation_data.csv
```

Output format:

* Same structure as input
* Labels replaced with predicted values (0 or 1)

---
