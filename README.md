**Adaptive Fraud Detection with Concept Drift Detection**

Demonstrates concept drift detection & automated model adaptation. 

---

**Project Overview**

Financial fraud patterns evolve over time - Attackers change tactics, transaction amounts shift, and new fraud vectors emerge. A static ML model trained once and left to run will silently degrade as the real world drifts away from its training distribution.

This project builds a **self-healing fraud detection system** that:

1. **Detects** when the data distribution has shifted (concept drift)
2. **Triggers** automated model retraining on fresh data
3. **Explains** which features changed most using SHAP
4. **Visualises** everything in a Streamlit dashboard

**Key Results**

| Metric        | Static Model | Adaptive Model | Improvement |
|---------------|-------------|----------------|-------------|
| Mean ROC-AUC  | 0.686       | **0.974**      | +0.288      |
| Mean F1 Score | 0.370       | **0.891**      | +0.521      |
| Avg Precision | 0.386       | **0.916**      | +0.530      |

**The adaptive model retrained 5 times** and maintained high performance throughout all three temporal periods (pre-drift, post-drift, recovery), while the static model completely failed during the drift period (F1 → 0.07).

---

**Project Structure**

```
fraud_drift_detection/
│
├── data/
│   └── creditcard_with_drift.csv       # Synthetic dataset with 3 drift periods
│
├── src/
│   ├── data_preprocessing.py           # Loading, scaling, streaming chunks
│   ├── baseline_model.py               # XGBoost + Random Forest baseline
│   ├── drift_detector.py               # ADWIN, DDM, Page-Hinkley detectors
│   ├── adaptive_retraining.py          # Sliding window + auto-retrain engine
│   └── explainability.py               # SHAP feature importance & drift shift
│
├── dashboard/
│   └── app.py                          # Streamlit monitoring dashboard
│
├── tests/
│   └── test_pipeline.py                # Unit tests (pytest)
│
├── models/                             # Saved model checkpoints (.pkl)
├── reports/                            # Generated plots, CSVs
│
├── run_experiment.py                   # 🚀 Main entry point — run this first
├── requirements.txt
└── README.md
```

---

**Quick Start Guide**

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full experiment
```bash
python run_experiment.py
```
This trains the baseline, streams all data, detects drift, retrains, and generates all plots in `reports/`.

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Run tests
```bash
python -m pytest tests/ -v
```

---

**Dataset**

A synthetic dataset with **40,000 transactions** across three temporal periods simulating realistic concept drift:

| Period       | Samples | Fraud Rate | What Changed                                |
|-------------|---------|-----------|----------------------------------------------|
| `pre_drift`  | 15,000  | 2.46%     | Fraud = low amounts, typical PCA features    |
| `post_drift` | 15,000  | 3.58%     | **Drift:** Fraud shifts to HIGH-value amounts; V1, V4 flip |
| `recovery`   | 10,000  | 2.98%     | Partial stabilisation of new fraud patterns  |

The dataset mimics the structure of the real-world [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (28 PCA-transformed features + Amount).

---

**Drift Detection Methods**

Three complementary detectors run in parallel:

### ADWIN (Adaptive Windowing)
- Maintains two sub-windows and tests whether their means differ significantly
- Fast: O(log n) per update
- Good for: abrupt and gradual drift

### DDM (Drift Detection Method)
- Monitors the error rate mean (`p`) and standard deviation (`s`)
- Triggers warning when `p + s > p_min + 2 * s_min`
- Triggers drift when `p + s > p_min + 3 * s_min`
- Good for: error-rate based drift in classification

### Page-Hinkley Test
- Cumulative sum detector for persistent mean shifts
- Good for: detecting gradual monotonic drift in prediction errors

---

### Adaptive Retraining

When any detector fires:
1. The `AdaptiveRetrainer` accumulates recent samples in a **sliding window** (3,000 samples)
2. It fits a fresh `StandardScaler` and `XGBoost` on the buffered data
3. Before/after ROC-AUC is logged to `reports/retrain_log.csv`
4. The new model replaces the old one for subsequent predictions
5. Detectors are reset to begin monitoring the new baseline

---

## SHAP Explainability

After the experiment, SHAP TreeExplainer identifies which features changed most between the pre-drift and post-drift models:

- `Amount` — shifted from low to ~5× higher importance (fraud moved to high-value txns)
- `V21`, `V22`, `V15` — became less important as fraud patterns changed
- `V12`, `V16` — became more important in the post-drift model

This is the **"explainable drift"** angle — not just detecting that drift happened, but explaining *why* and *which features* drove it.

---

## Dashboard Pages (Streamlit)

| Page                    | What it shows                                          |
|-------------------------|-------------------------------------------------------|
| 📊 Overview Dashboard   | KPIs, F1 comparison chart, period distribution        |
| 📈 Performance Comparison | Selectable metric, smoothed comparison, summary table |
| 🚨 Drift Events          | Timeline, detector breakdown, full event table        |
| 🔄 Retraining Log        | Before/after ROC-AUC bar chart, retrain details       |
| 📋 Data Explorer         | Feature distributions by drift period                 |

---

## Tech Stack

| Component           | Tool/Library                                       |
|---------------------|----------------------------------------------------|
| ML Models           | XGBoost, scikit-learn, imbalanced-learn            |
| Drift Detection     | Custom ADWIN, DDM, Page-Hinkley (River-compatible) |
| Explainability      | SHAP (TreeExplainer)                               |
| Dashboard           | Streamlit + Plotly                                 | 
| Experiment Tracking | MLflow-ready structure                             |
| Testing             | pytest                                             |
| Data                | pandas, numpy, scikit-learn synthetic              |

-------------------------------------------------------------------------------------------------

# Algorithm Mechanics

Raw stream -------> XGBoost ---------------> Drift Detectors ------> Auto-train -------> SHAP

## Part 1: Data stream (raw)
In the real world, a bank doesn't give you all transactions at once. They arrive one by one, continuously, like a river:

t=0    → transaction 1  (card swipe at Woolworths, $23.50)

t=1    → transaction 2  (online purchase, $340.00)

t=2    → transaction 3  (ATM withdrawal, $200.00)

...

t=∞    → keeps flowing forever

A **data stream** is this continuous, ordered sequence of transactions arriving in real time. The challenge is to make a fraud/not-fraud decision immediately.  

For this algorithm, we simulate streaming by splitting 40,000 transactions into 100 chunks of 400 using `get_streaming_chunks()`. Each chunk represents one batch of transactions arriving together — like one minute of activity at a bank.

Our dataset is based on the famous Kaggle Credit Card Fraud dataset collected from European cardholders in September 2013 (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Our dataset has 32 columns: V1-V28, Amount, Class, Time, Period
**V1-V28**: are the result of **PCA transformation** applied to the original transaction features. The bank applies PCA for privacy reasons. They cannot publish raw customer data publicly, but they can publish the mathematically transformed version which is anonymous. 

So, what PCA (Principal Component Analysis) does - it takes the original features — merchant name, location, card type, customer age, device ID, IP address, etc. — and mathematically compresses them into a smaller set of numbers that capture the same information.

<img width="1440" height="520" alt="image" src="https://github.com/user-attachments/assets/96188be6-9ed2-4255-81a9-f6e8ad6ee491" /> 

where, 
X = original feature vector (merchant, location, device, etc.)

W = PCA weight matrix (learned from data)

V = compressed feature vector [V1, V2, ... V28]

Each Vᵢ is a weighted combination of ALL original features:

V1 = w₁₁×merchant + w₁₂×age + w₁₃×device + ...

V2 = w₂₁×merchant + w₂₂×age + w₂₃×device + ...

...

PCA orders the components so that V1 captures the most variance in the data, V2 the second most, and so on down to V28 which captures the least. This is why V1–V5 tend to be the strongest fraud predictors. So, even though we cannot see the original labels, we can infer what each group probably represents from the SHAP analysis: 

| Feature        | Likely Represents                   | Fraud Signal                 |
|----------------|-------------------------------------|-------------------------------
| V1,V2,V3       | Transaction behaviour patterns      | Very strong - top predictors |
| V4,V5,V6       | Merchant/local patterns             | Strong                       |
| V7-V14         | Customer history patterns           | Moderate                     |
| V15-V21        | Time/frequency patterns             | Weaker                       |
| V22-V28        | Residual minor patterns             | Weakest                      |
|----------------|-------------------------------------|------------------------------|

**How concept drift affects the features** 
The model learned "fraud looks like small amounts + pattern A" — but fraudsters changed tactics, so now "fraud looks like large amounts + pattern B" — the model is now blind to the new fraud. 

Three types of drift

### 1. Feature value drift — the raw values shift 
Amount before:  fraud = ~$50

Amount after:   fraud = ~$500

→ The feature itself changed distribution

### 2. Feature importance drift — a feature becomes useless or newly important 
V21 before:  SHAP = 0.73  (very useful)

V21 after:   SHAP = 0.02  (useless)

→ The feature still exists but no longer predicts fraud

### 3. Relationship drift — the direction of a feature flips
V1 before:  high positive value → likely fraud

V1 after:   high negative value → likely fraud

→ The model's learned rule is now backwards


#



