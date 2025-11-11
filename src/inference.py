import numpy as np

def predict(model, scaler, features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    confidence = float(np.max(probabilities))
    return prediction, confidence, probabilities
