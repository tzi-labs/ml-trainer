
import gzip
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
records = []
with gzip.open("data/sample.json.gz", "rt", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
df["ts"] = pd.to_datetime(df["ts"], unit="ms")
df["browser"] = df["bn"].astype("category").cat.codes
df["label"] = df["category"].astype("category").cat.codes

X = df[["value", "browser"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")
print("Model trained and saved to models/model.pkl")
