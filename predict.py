# predict.py

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

model = joblib.load("models/sklearn_model.joblib")

data = fetch_california_housing()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"[Docker] R² Score: {score:.4f}")
