import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('ipl_data.csv')

columns = [
    'venue',
    'bat_team',
    'bowl_team',
    'runs',
    'wickets',
    'overs',
    'runs_last_5',
    'wickets_last_5',
    'striker',
    'non-striker',
    'total'
]

df = df[columns]
df = df[df['overs'] >= 5.0]

X = df.drop('total', axis=1)
y = df['total']

categorical_cols = ['venue', 'bat_team', 'bowl_team']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.4f}')
print(f'R2 Score: {r2:.4f}')

# Save Model with zlib compression (level 3 for fast loading and small size)
joblib.dump(model, 'score_predictor.pkl', compress=3)
print("Model saved successfully as score_predictor.pkl")
