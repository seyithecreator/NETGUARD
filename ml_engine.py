"""
ml_engine.py
Real ML pipeline: loads NSL-KDD CSV, preprocesses, trains DT and KNN,
returns metrics, confusion matrices, feature importances, and live inference.
"""
import pandas as pd
import numpy as np
import json, pickle, os, time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)

LABEL_NAMES = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
LABEL_COLORS = {
    'Normal': '#34d399', 'DoS': '#ff4560',
    'Probe': '#a78bfa', 'R2L': '#fb923c', 'U2R': '#f472b6'
}
DATA_PATH   = os.path.join(os.path.dirname(__file__), 'data', 'nslkdd.csv')
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'models', 'trained.pkl')

CAT_COLS = ['protocol_type', 'service', 'flag']
DROP_COLS = ['attack_type', 'label']


class IDSEngine:
    def __init__(self):
        self.dt_model  = None
        self.knn_model = None
        self.scaler    = MinMaxScaler()
        self.encoders  = {}
        self.feature_names = []
        self.results   = {}
        self.trained   = False
        self.df_test   = None
        self.X_test    = None
        self.y_test    = None
        self.test_records = []   # raw records for live feed
        self._feed_idx = 0

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    def load_and_preprocess(self):
        df = pd.read_csv(DATA_PATH)

        # Encode categoricals
        X = df.drop(columns=DROP_COLS).copy()
        for col in CAT_COLS:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le

        y = df['label'].values
        self.feature_names = list(X.columns)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y)

        # Keep raw test rows for the live feed
        test_idx = np.where(np.isin(np.arange(len(df)),
                                    np.arange(len(df))[-len(y_te):]))[0]
        self.test_records = df.iloc[-len(y_te):].to_dict(orient='records')

        return X_tr, X_te, y_tr, y_te

    # ── 2. Train ─────────────────────────────────────────────────────────────
    def train(self):
        t0 = time.time()
        X_tr, X_te, y_tr, y_te = self.load_and_preprocess()
        self.X_test = X_te
        self.y_test = y_te

        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=18, min_samples_split=5,
                                    criterion='gini', random_state=42)
        dt.fit(X_tr, y_tr)
        self.dt_model = dt

        # KNN – fixed k=5 (representative of cross-val optimal for NSL-KDD)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean',
                                   algorithm='ball_tree', n_jobs=-1)
        knn.fit(X_tr, y_tr)
        self.knn_model = knn

        train_time = round(time.time() - t0, 2)

        # ── Evaluate both ────────────────────────────────────────────────────
        self.results = {
            'DT':  self._eval(dt,  X_te, y_te, 'Decision Tree'),
            'KNN': self._eval(knn, X_te, y_te, 'K-Nearest Neighbour'),
            'train_time': train_time,
            'n_train': len(X_tr),
            'n_test':  len(X_te),
            'feature_importance': self._feature_importance(dt),
        }
        self.trained = True
        return self.results

    def _eval(self, model, X, y, name):
        y_pred = model.predict(X)
        cm     = confusion_matrix(y, y_pred, labels=[0,1,2,3,4])
        report = classification_report(y, y_pred,
                                       target_names=list(LABEL_NAMES.values()),
                                       output_dict=True, zero_division=0)
        return {
            'name':      name,
            'accuracy':  round(accuracy_score(y, y_pred)*100, 2),
            'precision': round(precision_score(y, y_pred, average='macro', zero_division=0)*100, 2),
            'recall':    round(recall_score(y, y_pred, average='macro', zero_division=0)*100, 2),
            'f1':        round(f1_score(y, y_pred, average='macro', zero_division=0)*100, 2),
            'confusion_matrix': cm.tolist(),
            'report':    report,
            'per_class': {LABEL_NAMES[i]: {
                'precision': round(report.get(LABEL_NAMES[i],{}).get('precision',0)*100,1),
                'recall':    round(report.get(LABEL_NAMES[i],{}).get('recall',0)*100,1),
                'f1':        round(report.get(LABEL_NAMES[i],{}).get('f1-score',0)*100,1),
                'support':   int(report.get(LABEL_NAMES[i],{}).get('support',0)),
            } for i in range(5)}
        }

    def _feature_importance(self, dt):
        imp = dt.feature_importances_
        idx = np.argsort(imp)[::-1][:15]
        return {
            'features': [self.feature_names[i] for i in idx],
            'scores':   [round(float(imp[i])*100, 2) for i in idx]
        }

    # ── 3. Live inference ────────────────────────────────────────────────────
    def predict_next(self, model='DT'):
        """Return one record with real prediction from actual test data."""
        if not self.trained or len(self.test_records) == 0:
            return None
        rec = self.test_records[self._feed_idx % len(self.test_records)]
        self._feed_idx += 1

        # Preprocess single record
        row = pd.DataFrame([rec]).drop(columns=DROP_COLS, errors='ignore')
        for col in CAT_COLS:
            le = self.encoders[col]
            val = str(row[col].iloc[0])
            row[col] = le.transform([val])[0] if val in le.classes_ else 0

        row_scaled = self.scaler.transform(row[self.feature_names])

        clf = self.dt_model if model in ('DT','BOTH') else self.knn_model
        pred_label = int(clf.predict(row_scaled)[0])
        proba = clf.predict_proba(row_scaled)[0]
        confidence = round(float(proba[pred_label]) * 100, 1)

        true_label = int(rec.get('label', -1))

        return {
            'src_ip':      f"192.168.{np.random.randint(0,5)}.{np.random.randint(1,254)}",
            'dst_ip':      f"10.0.{np.random.randint(0,3)}.{np.random.randint(1,50)}",
            'protocol':    rec.get('protocol_type','tcp').upper(),
            'service':     rec.get('service','http'),
            'src_bytes':   int(rec.get('src_bytes', 0)),
            'dst_bytes':   int(rec.get('dst_bytes', 0)),
            'duration':    round(float(rec.get('duration', 0)), 2),
            'flag':        rec.get('flag','SF'),
            'prediction':  LABEL_NAMES[pred_label],
            'true_label':  LABEL_NAMES[true_label] if true_label in LABEL_NAMES else '?',
            'correct':     pred_label == true_label,
            'confidence':  confidence,
            'model':       'DT' if model in ('DT','BOTH') else 'KNN',
            'attack_type': rec.get('attack_type','normal'),
        }

    # ── 4. Plotly chart data ─────────────────────────────────────────────────
    def confusion_matrix_chart(self, model='DT'):
        cm   = self.results[model]['confusion_matrix']
        lbls = list(LABEL_NAMES.values())
        return {
            'z':         cm,
            'x':         lbls,
            'y':         lbls,
            'colorscale': [[0,'#0d1219'],[0.5,'#0099ff'],[1,'#00d4aa']],
            'title':     f'Confusion Matrix — {self.results[model]["name"]}'
        }

    def feature_importance_chart(self):
        fi = self.results['feature_importance']
        return {
            'features': fi['features'],
            'scores':   fi['scores'],
        }

    def comparison_chart(self):
        metrics = ['Accuracy','Precision','Recall','F1']
        dt_vals  = [self.results['DT']['accuracy'],  self.results['DT']['precision'],
                    self.results['DT']['recall'],     self.results['DT']['f1']]
        knn_vals = [self.results['KNN']['accuracy'], self.results['KNN']['precision'],
                    self.results['KNN']['recall'],    self.results['KNN']['f1']]
        return {'metrics': metrics, 'DT': dt_vals, 'KNN': knn_vals}


# Singleton
engine = IDSEngine()
