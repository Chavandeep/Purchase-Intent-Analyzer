# 🛒 Purchase Intent Predictor

A machine learning system that predicts whether an e-commerce visitor is likely to make a purchase — based purely on their browsing behavior — and recommends targeted campaigns for each user segment.

**[Live Demo →](https://your-app.streamlit.app)** *(replace with your Streamlit URL once deployed)*

---

## Result
<img width="1920" height="2466" alt="image" src="https://github.com/user-attachments/assets/9e28aa87-4785-4f82-9739-215762afd87f" />

---

## What it does

Most e-commerce stores treat every visitor the same. This tool segments them by purchase intent so you can run the right campaign on the right person:

| Segment | Score | Action |
|---------|-------|--------|
| 🟢 High intent | ≥ 65 | Loyalty points — they're likely buying anyway |
| 🟡 Medium intent | 35–64 | Social proof or limited stock nudge |
| 🔴 Low intent | < 35 | Exit-intent discount coupon |

Upload a CSV of session data → get back a scored, segmented list ready to plug into Mailchimp, Klaviyo, or your ad manager.

---

## How it works

1. Export session data from Google Analytics, Mixpanel, or any analytics platform as a CSV
2. Upload it to the dashboard
3. The trained neural network scores every session in seconds
4. Download the results — each row now has an `intent_score` and `segment` column

No data science knowledge needed.

---

## Model

- **Architecture:** 5-layer neural network (Keras / TensorFlow)
- **Dataset:** UCI Online Shoppers Purchasing Intention Dataset — 12,330 real e-commerce sessions
- **Accuracy:** ~88% on held-out test set
- **Features used:** Product page dwell time, page value score, bounce rate, exit rate, visitor type, session context (month, weekend, special day proximity)

---

## Tech stack

- Python, Keras, TensorFlow, scikit-learn
- Pandas, NumPy
- Streamlit
- Deployed on Streamlit Cloud

---

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/Chavandeep/Purchase-Intent-Predictor.git
cd Purchase-Intent-Predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train and save the model** *(run once)*
```bash
python train_and_save_model.py
```
Requires `data/dataset.csv` — download the UCI dataset from [here](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).

**4. Launch the dashboard**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Expected CSV columns

Your upload should contain these columns:

```
Administrative_Duration, Informational_Duration, ProductRelated_Duration,
BounceRates, ExitRates, PageValues, SpecialDay,
Month, Browser, VisitorType, Weekend
```

Optional columns (ignored if present):
```
Administrative, Informational, ProductRelated,
OperatingSystems, Region, TrafficType,
Revenue  ← include this to see model accuracy on your data
```

---

## Project structure

```
Purchase-Intent-Predictor/
├── app.py                    ← Streamlit dashboard
├── train_and_save_model.py   ← Train and save model (run once)
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv           ← UCI dataset (not committed)
└── model/
    ├── model.h5
    ├── scaler.pkl
    ├── encoder.pkl
    └── label_encoders.pkl
```

---

## Limitations & future scope

- **Batch only right now** — users upload a weekly CSV export. A future version would add a live JS tracker + API endpoint to score visitors in real time as they browse.
- **Column mapping** — currently expects UCI dataset column names. A column mapper would make it compatible with any analytics export format.
- **Model retraining** — accuracy improves over time when retrained on site-specific data with real purchase outcomes.

---

## Dataset

[UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) — Sakar et al., 2018.
