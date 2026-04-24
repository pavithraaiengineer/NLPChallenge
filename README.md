# Fake News Classifier

Binary NLP classification project distinguishing fake (0) from real (1) news headlines using a progression of models from classical baselines to a DistilBERT ensemble.

**Team:** Charleson · Pavithra · Sara

---

## Results

| Model | Validation Accuracy |
|---|---|
| Logistic Regression (BoW) | 94.3% |
| Naïve Bayes (BoW) | 94.09% |
| Linear SVC (TF-IDF) | 95.37% |
| Random Forest (TF-IDF) | 91.66% |
| MiniLM Embeddings + LR | 96.61% |
| DistilBERT Embeddings (BiLSTM) | 96.61% |
| DistilBERT Fine-Tuned | 98.68% |
| **Ensemble (Final)** | **98.79%** |

Final model metrics: Accuracy **98.79%** · Precision **0.99** · Recall **0.99** 

---

## Repo Structure

```
├── preProc.py                  # Text cleaning, stopword removal, stemming, lemmatisation
├── model1.ipynb                # Baseline classifiers (BoW, TF-IDF, 4 sklearn models)
├── model2.ipynb                # + DistilBERT embeddings, precision/recall/ROC, plots
├── training_data_lowercase.csv # Tab-separated dataset (label, title)
└── README.md
```

---

## Dataset

- **Size:** 34,152 samples
- **Format:** Tab-separated CSV with columns `label` and `title`
- **Labels:** `0` = Fake, `1` = Real
- **Pre-processing:** Dataset is already lowercased
- **Split:** 80% train / 20% validation (`random_state=42`)
- **Class balance:** Approximately 50/50 — no class weighting required

---

## Preprocessing Findings

We tested stopword removal, stemming (PorterStemmer) and lemmatisation (WordNetLemmatizer) against raw cleaned text. All three preprocessing steps **reduced** model performance, but combining raw cleaned text with stemming led to the best results in classic models.

The reason: news headlines are already short, so every word carries signal. Function words and inflectional patterns contribute stylistic cues that distinguish fake from real writing. Removing or normalising them discards information the models rely on. DistilBERT's WordPiece tokeniser also handles morphology internally, so pre-lemmatising before feeding text to BERT removes information the model was pretrained to use.

**Conclusion:** `clean_text` (punctuation removed, lowercased) fed directly to all vectorisers and DistilBERT gave the best results across all model families.

---

## Model Progression

### 1 — Baseline Classifiers
Bag-of-Words and TF-IDF vectorisers (`max_features=5000`) paired with four sklearn classifiers. Best result: **Linear SVC + combination of raw and stemm with TF-IDF → 95.37%**.

### 2 — Sentence Embeddings
Static dense vectors from `all-MiniLM-L6-v2` (384-d) and `distilbert-base-uncased` (768-d) used as drop-in replacements for BoW/TF-IDF. No training — weights frozen. Vectors extracted by averaging `last_hidden_state` across the token dimension and wrapped in `csr_matrix` for sklearn compatibility. Best: **DistilBERT Embeddings + LR → 96.61%**.

### 3 — DistilBERT Fine-Tuning
Full fine-tune of `distilbert-base-uncased` on the training set using HuggingFace `Trainer` with AdamW, learning rate `2e-5`, batch size 16, EarlyStopping on eval loss, 2 epochs. Seed 42 outperformed Seed 0. Result: **98.68%**.

### 4 — Ensemble (Final Model)
Soft-vote ensemble combining probability scores from:
- DistilBERT static embeddings → Logistic Regression
- DistilBERT fine-tuned → probability outputs
- Calibrated Linear SVC (CalibratedClassifierCV required for valid probability scores)

Result: **98.79% accuracy — most stable configuration across seeds.

---

## Setup

### Requirements

```bash
pip install pandas scikit-learn matplotlib scipy
pip install transformers torch
pip install sentence-transformers   # for MiniLM experiments
```

A **GPU is strongly recommended** for DistilBERT fine-tuning. The notebooks were run on Google Colab with a T4 GPU. Encoding 34k samples takes approximately 20 minutes per run on CPU.

### Running

1. Place `training_data_lowercase.csv` in the project root
2. Ensure `preProc.py` is in the same directory
3. Run `Baseline.ipynb` for baseline + embedding experiments
4. Run `Final_EMB+FT+ENS.ipynb` for the full pipeline including DistilBERT and ensemble

---

## preProc.py Functions

| Function | Description |
|---|---|
| `normalize_text(text)` | Removes punctuation and special characters, lowercases |
| `remove_stopwords(text)` | Removes NLTK English stopwords |
| `tokens_stemm(text)` | PorterStemmer tokenisation |
| `tokens_lemm(text)` | WordNetLemmatizer tokenisation |

---

## Key Implementation Notes

- `MultinomialNB` is **skipped** for embedding experiments — it requires non-negative inputs and DistilBERT vectors can be negative
- `LinearSVC` has no `predict_proba` — `CalibratedClassifierCV` wraps it for valid probability outputs used in ROC-AUC and ensembling
- DistilBERT embedding uses `mean` pooling over `last_hidden_state` (shape: `batch × tokens × 768`) to produce one 768-d vector per sample
- Embeddings are processed in batches of 16 to avoid OOM errors
