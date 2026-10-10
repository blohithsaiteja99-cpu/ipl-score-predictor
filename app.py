import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# Load ML Model
MODEL_PATH = os.path.join(BASE_DIR, 'score_predictor.pkl')
model = joblib.load(MODEL_PATH)

# CricAPI Key
API_KEY = os.getenv("CRICAPI_KEY", "2538a5f-cf9a-4236-a847-cf945518d0c0")
LIVE_URL = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}&offset=0"

@app.route('/')
def home():
    live_matches = []
    try:
        response = requests.get(LIVE_URL, timeout=5)
        data = response.json()
        if data.get('status') == 'success':
            live_matches = data.get('data', [])
    except Exception:
        live_matches = []

    return render_template(
        'index.html',
        live_matches=live_matches
    )

@app.route('/predict', methods=['POST'])
def predict():
    venue = request.form['venue']
    bat_team = request.form['bat_team']
    bowl_team = request.form['bowl_team']

    runs = int(request.form['runs'])
    wickets = int(request.form['wickets'])
    overs = float(request.form['overs'])
    runs_last_5 = int(request.form['runs_last_5'])
    wickets_last_5 = int(request.form['wickets_last_5'])
    striker = int(request.form['striker'])
    non_striker = int(request.form['non_striker'])

    input_df = pd.DataFrame({
        'venue': [venue],
        'bat_team': [bat_team],
        'bowl_team': [bowl_team],
        'runs': [runs],
        'wickets': [wickets],
        'overs': [overs],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5],
        'striker': [striker],
        'non-striker': [non_striker]
    })

    prediction = round(model.predict(input_df)[0])

    live_matches = []
    try:
        response = requests.get(LIVE_URL, timeout=5)
        data = response.json()
        if data.get('status') == 'success':
            live_matches = data.get('data', [])
    except Exception:
        live_matches = []

    return render_template(
        'index.html',
        prediction_text=f'Predicted Score : {prediction}',
        live_matches=live_matches
    )

if __name__ == '__main__':
    app.run(debug=True)