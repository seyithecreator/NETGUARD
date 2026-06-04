"""
app.py  —  NetGuard IDS Flask backend
Run:  python app.py
"""
from flask import Flask, render_template, jsonify, request
from ml_engine import engine
import threading, time, random

app = Flask(__name__)

# ── Train on startup in background ───────────────────────────────────────────
training_status = {'done': False, 'error': None}

def train_background():
    try:
        engine.train()
        training_status['done'] = True
    except Exception as e:
        training_status['error'] = str(e)

t = threading.Thread(target=train_background, daemon=True)
t.start()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'trained': training_status['done'],
        'error':   training_status['error']
    })

@app.route('/api/metrics')
def metrics():
    if not training_status['done']:
        return jsonify({'ready': False}), 202
    r = engine.results
    return jsonify({
        'ready': True,
        'n_train': r['n_train'],
        'n_test':  r['n_test'],
        'train_time': r['train_time'],
        'DT':  {k: r['DT'][k]  for k in ['accuracy','precision','recall','f1','per_class']},
        'KNN': {k: r['KNN'][k] for k in ['accuracy','precision','recall','f1','per_class']},
    })

@app.route('/api/chart/confusion/<model>')
def confusion(model):
    if not training_status['done']:
        return jsonify({'ready': False}), 202
    return jsonify(engine.confusion_matrix_chart(model.upper()))

@app.route('/api/chart/features')
def features():
    if not training_status['done']:
        return jsonify({'ready': False}), 202
    return jsonify(engine.feature_importance_chart())

@app.route('/api/chart/comparison')
def comparison():
    if not training_status['done']:
        return jsonify({'ready': False}), 202
    return jsonify(engine.comparison_chart())

@app.route('/api/predict')
def predict():
    if not training_status['done']:
        return jsonify({'ready': False}), 202
    model = request.args.get('model', 'DT')
    rec   = engine.predict_next(model)
    if rec is None:
        return jsonify({'ready': False}), 202
    return jsonify(rec)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
