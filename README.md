# NetGuard IDS — Real-Time Intrusion Detection Dashboard

A fully functional IDS prototype using Decision Tree and KNN classifiers
trained on NSL-KDD network traffic data, served via a Flask web dashboard.

## Stack
- **Python 3.10+**
- **scikit-learn 1.3** — DT & KNN classifiers, MinMaxScaler, preprocessing
- **pandas 2.0** — dataset loading and feature engineering
- **numpy 1.25** — array operations throughout the pipeline
- **Flask** — lightweight web server and API layer
- **Plotly** — interactive confusion matrices, feature importance, model comparison charts
  *(replaces static Matplotlib/Seaborn — same data, interactive in the browser)*

## Project Structure
```
netguard/
├── app.py              ← Flask server + API routes
├── ml_engine.py        ← Full ML pipeline (train, evaluate, predict)
├── requirements.txt
├── data/
│   └── nslkdd.csv      ← NSL-KDD dataset (generated / replace with real)
├── models/             ← Serialised model artefacts (auto-created)
└── templates/
    └── index.html      ← Dashboard UI
```

## Setup & Run

```bash
# 1. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open browser
# → http://localhost:5050
```

## Using the Real NSL-KDD Dataset
Replace `data/nslkdd.csv` with the real dataset from:
https://www.unb.ca/cic/datasets/nsl.html

The expected columns are the standard 41 NSL-KDD features plus:
- `attack_type` (string label e.g. 'neptune', 'normal')
- `label` (integer: 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)

## Dashboard Pages

### Live Monitor
- Real-time packet feed classified by trained DT or KNN model
- Confidence scores, true vs predicted labels, per-category counters
- Switch between DT / KNN / BOTH classifiers in real time
- Filter and search the alert log

### Analytics
- Confusion matrices for both models (interactive Plotly heatmaps)
- Model comparison bar chart (Accuracy, Precision, Recall, F1)
- Top-15 feature importances from Decision Tree
- Per-class precision / recall / F1 table for both models
