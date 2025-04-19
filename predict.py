import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/model.pkl")

# Match training feature names exactly
X_input = pd.DataFrame([{
    "browser": 1,
    "is_mobile": 1,
    "ref_from_google": 1,
    "vp_w": 390,
    "vp_h": 700
}])

# Predict
prediction = model.predict(X_input)
probability = model.predict_proba(X_input)[0][1]

print("Predicted engagement label:", prediction[0])
print("Engagement probability:", round(probability, 4))
