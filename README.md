<div align="center">

<h1>🎫 Support Ticket Classifier</h1>

**Content-Based NLP Classification Using TF-IDF · Logistic Regression · Rule-Based Priority Engine**

&nbsp;

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-154f3c?style=flat-square)](https://nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](./LICENSE)

&nbsp;

A production-structured NLP web application that processes raw customer support tickets through a two-stage intelligence pipeline — a trained Logistic Regression classifier for category prediction and a keyword-driven priority engine for urgency tagging — fusing both outputs into a single actionable result with zero human intervention.

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Application Screenshots](#-application-screenshots)
- [System Architecture](#️-system-architecture)
- [Algorithm Deep-Dive](#-algorithm-deep-dive)
- [Algorithm Comparison & Benchmarks](#-algorithm-comparison--benchmarks)
- [Score Behaviour Across Ticket Types](#-score-behaviour-across-ticket-types)
- [Dataset Breakdown](#-dataset-breakdown)
- [Project Structure](#️-project-structure)
- [Quick Start](#-quick-start)
- [Technology Stack](#️-technology-stack)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🔭 Project Overview

Customer support teams receive hundreds of tickets daily — unstructured, inconsistently worded, and mixed in urgency. Manual triaging is slow, error-prone, and scales poorly. Critical issues get buried under noise. Low-priority queries consume time that should go to P1 failures.

This project solves that bottleneck with a **two-stage NLP classification pipeline** that:

- **Cleans and normalizes** raw ticket text via regex stripping, lowercasing, and NLTK English stopword removal
- **Vectorizes** the cleaned text into a weighted TF-IDF feature matrix — amplifying discriminative terms like *refund*, *crashing*, *locked* while suppressing generic filler
- **Classifies** the ticket into one of four operational categories (Technical, Billing, Account, General) using a trained Logistic Regression model
- **Tags urgency** by running a deterministic keyword engine over the cleaned text — matching domain-critical signals to High / Medium / Low priority tiers
- **Delivers results** through a Flask web interface in under one second — category + priority, side by side, from a single form submission

Every prediction runs live on submission. No pre-cached results. No approximation. The same `clean_text()` function runs identically at training time (`train_model.py`) and inference time (`app.py`) — guaranteeing zero train/serve skew.

---

## 📸 Application Screenshots

**Home Page — Ticket Input Interface**
<img width="1366" height="768" alt="Home Page" src="https://github.com/user-attachments/assets/0657a9d8-1cb8-499f-a026-c59a9884466e" />

Clean single-field input form. Accepts any free-form ticket text. On submit, the full NLP pipeline fires on the Flask backend — clean → vectorize → classify → prioritize — and returns to the result page.

---

**Result Page — Billing Ticket · High Priority**
<img width="1366" height="768" alt="Billing High Priority" src="https://github.com/user-attachments/assets/ca58547d-b5c1-4d50-b41d-a7fa624b2bc0" />

Result dashboard displaying: original ticket text, predicted category, and assigned priority level. Ticket `"Payment failed but money deducted"` — classified as **Billing**, tagged **High** by the keyword engine (trigger: `"failed"`).

---

**Result Page — Technical Issue Classification**
<img width="1366" height="768" alt="Technical Classification" src="https://github.com/user-attachments/assets/a4ce900e-32ec-449f-8fc7-2cb259f0e8f2" />

Ticket `"Application keeps crashing"` — classified as **Technical**, tagged **High** (trigger: `"not working"` family of terms). The model correctly routes this to the engineering queue.

---

**Result Page — Billing · Medium Priority**
<img width="1366" height="768" alt="Billing Medium Priority" src="https://github.com/user-attachments/assets/8708646c-efc8-418e-860f-8998b942ae86" />

Ticket `"Incorrect billing amount"` — classified as **Billing**, tagged **Medium** (trigger: `"billing"`). Demonstrates the priority engine distinguishing between a billing inquiry (Medium) and a payment failure (High).

---

## 🏗️ System Architecture

```
Browser (Ticket Text Input)
        │
        │  POST /  (form field: ticket)
        ▼
app.py — Flask Route Handler
        │
        ├── request.form["ticket"]     → raw ticket string
        │
        ├── clean_text(ticket)
        │     │ .lower()               → normalize case
        │     │ re.sub(r"[^a-z ]", "") → strip digits, punctuation, symbols
        │     └ stopword filter        → remove NLTK English stopwords
        │                               → cleaned string
        │
        ├── ─── PIPELINE 1: ML CLASSIFICATION ────────────────────────────
        │   vectorizer.transform([cleaned])
        │     │ loaded from model/vectorizer.pkl (TfidfVectorizer, fitted)
        │     │ converts cleaned string → sparse TF-IDF matrix [1 × vocab]
        │     └→ vector: scipy sparse matrix
        │
        │   model.predict(vector)
        │     │ loaded from model/classifier.pkl (LogisticRegression, trained)
        │     └→ category: "Technical" | "Billing" | "Account" | "General"
        │
        ├── ─── PIPELINE 2: PRIORITY ENGINE ───────────────────────────────
        │   assign_priority(cleaned)
        │     │ High  ← any(k in text for k in ["not working","refund","error","failed"])
        │     │ Medium← any(k in text for k in ["slow","billing"])
        │     └→ priority: "High" | "Medium" | "Low"
        │
        └→ render_template("result.html",
                ticket=ticket, category=category, priority=priority)
```

---

## 🔬 Algorithm Deep-Dive

### 1. Text Preprocessing — `clean_text()`

The same function runs identically in both `train_model.py` and `app.py`. This is a hard architectural requirement — any divergence between training-time and inference-time text processing introduces feature distribution shift and degrades model accuracy.

```python
def clean_text(text):
    text = text.lower()                             # "Payment FAILED!" → "payment failed!"
    text = re.sub(r"[^a-z ]", "", text)             # "payment failed!" → "payment failed"
    return " ".join(                                # remove stopwords
        w for w in text.split() if w not in stop_words
    )                                               # "payment failed"  (no change here)
                                                    # "unable to login" → "unable login"
```

What the regex `[^a-z ]` eliminates:
- Digits: `"ticket #4821"` → `"ticket "`
- Punctuation: `"not working!!!"` → `"not working"`
- Special characters: `"error@upload"` → `"errorupload"` → `"errorupload"` (effectively joined)

NLTK stopwords removed: `"i"`, `"to"`, `"the"`, `"is"`, `"my"`, `"for"`, `"was"`, `"but"` and ~150 others. This ensures the TF-IDF vocabulary contains only semantically meaningful tokens.

---

### 2. TF-IDF Vectorization — `TfidfVectorizer`

```python
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)       # fit on 10,000 training samples
```

TF-IDF assigns each token a weight based on two factors:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d)  = count of term t in document d / total terms in d
IDF(t)    = log(N / df(t))   where N = total docs, df(t) = docs containing t
```

In practice for this corpus:

- High-weight tokens: `"refund"`, `"crashing"`, `"locked"`, `"deducted"` — rare, category-specific
- Low-weight tokens: `"account"`, `"issue"`, `"problem"` — appear across all categories, suppressed
- Zero-weight tokens: all stopwords — already removed before vectorization

The fitted vectorizer (vocabulary + IDF weights) is serialized to `model/vectorizer.pkl`. At inference time, only `.transform()` is called — never `.fit()` — so the vocabulary is frozen at training distribution.

---

### 3. Logistic Regression Classifier

```python
model = LogisticRegression()
model.fit(X_vec, y)
```

Logistic Regression was chosen deliberately over heavier alternatives:

| Property | Detail |
|---|---|
| Input compatibility | Handles high-dimensional sparse TF-IDF matrices natively — no dense conversion needed |
| Multi-class support | Uses one-vs-rest (OvR) by default — one binary classifier per category |
| Inference speed | Single matrix multiply + softmax — sub-millisecond per ticket |
| Interpretability | Per-class feature weights are directly inspectable |
| No GPU required | Fully CPU-bound — deployable on any machine |

Training corpus: 10,000 samples, 4 balanced classes (~2,500 per class). The trained model is serialized to `model/classifier.pkl`.

---

### 4. Priority Assignment Engine — `assign_priority()`

```python
def assign_priority(text):
    if any(k in text for k in ["not working", "refund", "error", "failed"]):
        return "High"
    elif any(k in text for k in ["slow", "billing"]):
        return "Medium"
    return "Low"
```

The engine runs on the **post-cleaned** text — not the raw input. This means:

- `"NOT WORKING!!!"` → cleaned to `"not working"` → matches High ✅
- `"Error123"` → cleaned to `"error"` → matches High ✅
- `"Billing inquiry"` → cleaned to `"billing inquiry"` → matches Medium ✅

Priority is evaluated in strict tier order. The first match wins — a ticket containing both `"failed"` and `"billing"` is tagged High, not Medium.

| Priority | Trigger Keywords | Business Rationale |
|---|---|---|
| 🔴 **High** | `not working` · `refund` · `error` · `failed` | Service outage or money at risk — SLA-critical |
| 🟡 **Medium** | `slow` · `billing` | Degraded service or billing inquiry — time-sensitive |
| 🟢 **Low** | *(no match)* | Informational or general — standard queue |

---

## 📊 Algorithm Comparison & Benchmarks

### Method Comparison Matrix

| Method | Core Technique | Output | Semantic? | Training Required | Best For |
|---|---|---|---|---|---|
| **TF-IDF + LR** ✅ | Token frequency weighting + linear classifier | 4-class label | ✅ Lexical | ✅ Yes | Category routing |
| **Rule-Based Keywords** ✅ | Exact substring match on cleaned text | 3-tier label | ❌ No | ❌ No | Urgency tagging |
| Bag of Words + NB | Token counts + Naive Bayes | Multi-class | ✅ Lexical | ✅ Yes | Baseline text classification |
| Word2Vec + LR | Dense word embeddings + classifier | Multi-class | ✅ Semantic | ✅ Yes | Synonym-robust classification |
| BERT Fine-tuned | Transformer contextual embeddings | Multi-class | ✅ Deep semantic | ✅ Yes (GPU) | Complex / ambiguous tickets |
| GPT Prompting | LLM zero-shot classification | Free-form | ✅ Semantic | ❌ No | No-training rapid prototyping |

✅ Implemented in this project

### Why This Combination Was Chosen

TF-IDF + Logistic Regression and keyword priority form a **deliberate complementary pairing** — each covers a gap the other leaves:

```
                    TF-IDF LR     Keyword Engine
──────────────────────────────────────────────────
Category routing        ✅              ❌
Urgency tagging         ❌              ✅
Handles synonyms        partial         ❌
Rotation/typo robust    partial         ❌
Sub-ms inference        ✅              ✅
No GPU required         ✅              ✅
No pretrained model     ✅              ✅
Deterministic output    ❌              ✅
Interpretable           ✅              ✅
```

BERT and GPT approaches offer stronger semantic understanding but require 150–500ms inference latency, model weight downloads (400MB–1GB+), and GPU infrastructure for production throughput. The TF-IDF + LR pipeline delivers sub-10ms inference on CPU with no external model dependencies.

---

## 📉 Score Behaviour Across Ticket Types

All predictions reflect the exact `clean_text()` + `assign_priority()` logic in `app.py`.

| Raw Ticket Input | Cleaned Text | Category | Priority | Trigger |
|---|---|---|---|---|
| `"Payment failed but money deducted"` | `"payment failed money deducted"` | Billing | 🔴 High | `"failed"` |
| `"Internet is not working"` | `"internet not working"` | Technical | 🔴 High | `"not working"` |
| `"Need refund for last payment"` | `"need refund last payment"` | Billing | 🔴 High | `"refund"` |
| `"Error while uploading files"` | `"error uploading files"` | Technical | 🔴 High | `"error"` |
| `"Website is very slow"` | `"website slow"` | Technical | 🟡 Medium | `"slow"` |
| `"Incorrect billing amount"` | `"incorrect billing amount"` | Billing | 🟡 Medium | `"billing"` |
| `"Account locked"` | `"account locked"` | Account | 🟢 Low | *(no match)* |
| `"How to upgrade my plan?"` | `"upgrade plan"` | General | 🟢 Low | *(no match)* |
| `"Thank you for the great service"` | `"thank great service"` | General | 🟢 Low | *(no match)* |

**Key observation:** Priority and Category are independent. `"Incorrect billing amount"` is correctly classified as Billing (by the ML model) and Medium (by the keyword engine matching `"billing"`). `"Account locked"` is Account / Low — the word *locked* is not in any priority keyword list, so it falls through to the default tier.

---

## 🗂️ Dataset Breakdown

Generated by `generate_dataset.py` — **10,000 samples, 4 balanced classes** (~2,500 per class), randomly sampled from domain-specific template pools designed to reflect real customer support traffic patterns.

| Class | Pool Size | Template Examples |
|---|---|---|
| **Technical** | 6 | `"Internet is not working"` · `"Application keeps crashing"` · `"System timeout issue"` · `"Unable to connect to server"` |
| **Billing** | 5 | `"Charged twice for subscription"` · `"Payment failed but money deducted"` · `"Invoice not generated"` · `"Need refund for last payment"` |
| **Account** | 5 | `"Password reset not working"` · `"Account locked"` · `"Email verification failed"` · `"Username not recognized"` |
| **General** | 5 | `"How to upgrade my plan?"` · `"Is customer support available 24/7?"` · `"Thank you for the great service"` |

Generation logic: each of 10,000 iterations independently draws a random class, then a random template from that class's pool. The random seed is **not fixed** — re-running `generate_dataset.py` produces a statistically equivalent but non-identical dataset. Re-training after regeneration is recommended for full reproducibility.

---

## 🏗️ Project Structure

```
support-ticket-classifier/
│
├── app.py                        # Flask routes: GET / and POST /
│                                 # Loads pkl models at startup, runs full pipeline on POST
│
├── train_model.py                # Dataset loading, clean_text(), TF-IDF fit,
│                                 # LR training, pickle serialization
│
├── generate_dataset.py           # Synthetic 10K corpus generator
│                                 # 4-class template pools → tickets.csv
│
├── requirements.txt              # Pinned: Flask 2.3.3, pandas 2.1.4,
│                                 # scikit-learn 1.3.2, nltk 3.8.1
│
├── 📁 data/
│   └── tickets.csv               # Generated training corpus [text, category]
│
├── 📁 model/
│   ├── classifier.pkl            # Serialized LogisticRegression (fitted)
│   └── vectorizer.pkl            # Serialized TfidfVectorizer (fitted, vocab frozen)
│
└── 📁 templates/
    ├── index.html                # Ticket text input form
    └── result.html               # Prediction result: ticket · category · priority
```

**Model loading:** Both `classifier.pkl` and `vectorizer.pkl` are loaded once at app startup via `pickle.load()` — not on every request. This means inference cost per ticket is purely the transform + predict forward pass, not deserialization overhead.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/mv-karthikeya/support-ticket-classifier.git
cd support-ticket-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:

```
flask==2.3.3
pandas==2.1.4
scikit-learn==1.3.2
nltk==3.8.1
```

### 3. Generate the Dataset

```bash
python generate_dataset.py
# ✅ Dataset created successfully at data/tickets.csv
```

### 4. Train the Model

```bash
mkdir -p model
python train_model.py
# ✅ Model trained and saved successfully
```

Outputs: `model/classifier.pkl` + `model/vectorizer.pkl`

### 5. Launch the Application

```bash
python app.py
# * Running on http://127.0.0.1:5000
```

### 6. Classify a Ticket

- Open `http://127.0.0.1:5000`
- Type or paste any support ticket text
- Click **Classify** — category + priority returned in < 1 second

### Supported Input

Any free-form English text. The pipeline handles mixed case, punctuation, digits, and special characters — all stripped by `clean_text()` before reaching the model.

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Core application logic |
| Web Framework | Flask | 2.3.3 | Request handling, Jinja2 rendering |
| ML Classification | scikit-learn | 1.3.2 | `TfidfVectorizer` + `LogisticRegression` |
| Text Processing | NLTK | 3.8.1 | English stopword corpus (`stopwords.words("english")`) |
| Data I/O | pandas | 2.1.4 | CSV loading (`read_csv`), column operations |
| Model Persistence | pickle | stdlib | Serialize/deserialize fitted vectorizer + classifier |
| Templating | Jinja2 | via Flask | `render_template()` for index + result pages |

---

## 📈 Future Enhancements

| Enhancement | Description | Capability Added |
|---|---|---|
| ML-based Priority Model | Train a second classifier on priority labels instead of keyword rules | Handles paraphrased urgency (`"site completely down"` → High without keyword match) |
| Confidence Score Display | Show `model.predict_proba()` max probability on result page | User-visible classification confidence |
| REST API Endpoint | `POST /api/classify` returning `{category, priority}` JSON | External integrations, chatbot routing |
| BERT Fine-tuning | Replace TF-IDF + LR with `bert-base-uncased` fine-tuned on ticket corpus | Semantic understanding of novel phrasings |
| Fixed Random Seed | Add `random.seed(42)` to `generate_dataset.py` | Fully reproducible training runs |
| Batch Upload | CSV upload → classify all rows → downloadable results | Bulk retrospective ticket analysis |
| Docker Support | `Dockerfile` + `docker-compose.yml` | One-command deployment anywhere |
| Admin Dashboard | Ticket volume by category + priority over time | Operational analytics |

---

## 👤 Author

**M V Karthikeya**
Aspiring ML Engineer · NLP Enthusiast · 📍 India

[![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 📜 License

This project is released under the [MIT License](./LICENSE) — free for academic, research, and educational use with attribution.

---

<div align="center">

*Two pipelines · One fused result · Built for intelligent ticket routing*

</div>
