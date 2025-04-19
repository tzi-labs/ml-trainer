import os
import json
import gzip
import io
import pandas as pd
import boto3
import joblib
import datetime
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from botocore.config import Config

load_dotenv()

ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
SECRET_KEY = os.getenv("R2_SECRET_KEY")
ENDPOINT = os.getenv("R2_ENDPOINT")
BUCKET = os.getenv("R2_BUCKET")
PREFIX = os.getenv("R2_PREFIX")

config = Config(retries={'max_attempts': 3})

s3 = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    endpoint_url=ENDPOINT,
    config=config
)

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

# Load or initialize metadata
METADATA_FILE = "metadata/training_metadata.json"
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
else:
    metadata = {
        "processed_files": [],
        "model_version": "1.0",
        "last_training": None
    }

# Get list of already processed files
processed_files = set(item["key"] for item in metadata["processed_files"])
print(f"Found {len(processed_files)} already processed files")

# Get files from R2
response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
all_files = [obj["Key"] for obj in response.get("Contents", []) 
             if obj["Key"].endswith(".json.gz") and "processed" not in obj["Key"]]

# Filter for new files only
new_files = [key for key in all_files if key not in processed_files]
print(f"Found {len(new_files)} new files to process")

if not new_files:
    print("No new data to train on. Exiting.")
    exit(0)

# Process new files
records = []
for key in new_files:
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        obj_stream = response["Body"]._raw_stream
        raw_bytes = obj_stream.read()

        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz:
                lines = gz.read().decode("utf-8").splitlines()
        except gzip.BadGzipFile:
            lines = raw_bytes.decode("utf-8").splitlines()

        for line in lines:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
            
        # Mark file as processed
        metadata["processed_files"].append({
            "key": key,
            "processed_date": datetime.datetime.now().isoformat()
        })
        print(f"✅ Processed file: {key}")
        
    except Exception as e:
        print(f"⚠️ Skipping corrupted or unreadable file {key}: {e}")
        continue

# Prepare new data
if not records:
    print("No usable data found in new files. Exiting.")
    exit(0)

new_data_df = pd.DataFrame(records)
new_data_df["label"] = (new_data_df["ev"] == "pageclose").astype(int)
new_data_df["browser"] = new_data_df["bn"].astype("category").cat.codes
new_data_df["is_mobile"] = new_data_df["md"].astype(int)
new_data_df["ref_from_google"] = new_data_df["rl"].fillna("").str.contains("google").astype(int)
new_data_df["vp_w"] = new_data_df["vp"].str.split("x").str[0].astype(float)
new_data_df["vp_h"] = new_data_df["vp"].str.split("x").str[1].astype(float)

X_new = new_data_df[["browser", "is_mobile", "ref_from_google", "vp_w", "vp_h"]]
y_new = new_data_df["label"]

# Check if we have an existing model to update
MODEL_PATH = "models/model.pkl"
if os.path.exists(MODEL_PATH):
    print("Loading existing model for incremental update...")
    model = joblib.load(MODEL_PATH)
    
    # For RandomForestClassifier, we need to retrain on combined data
    # since it doesn't support true incremental learning
    
    # Option 1: Store previous training data and combine
    # This is memory intensive but most accurate
    if os.path.exists("models/previous_data.pkl"):
        X_prev, y_prev = joblib.load("models/previous_data.pkl")
        X_combined = pd.concat([X_prev, X_new])
        y_combined = pd.concat([y_prev, y_new])
        
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3)
        print(f"Training on combined dataset with {len(X_combined)} records")
        
    # Option 2: Use warm_start for RandomForest
    # Less accurate but more memory efficient
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3)
        model.warm_start = True
        print(f"Training with warm start on {len(X_new)} new records")
else:
    print("Creating new model...")
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3)
    print(f"Training new model on {len(X_new)} records")

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, MODEL_PATH)

# Save current data for future incremental training
joblib.dump((X_new, y_new), "models/previous_data.pkl")

# Update metadata
metadata["last_training"] = datetime.datetime.now().isoformat()
metadata["model_version"] = str(float(metadata["model_version"]) + 0.1)

# Save updated metadata
with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Engagement prediction model (v{metadata['model_version']}) trained and saved.")
print(f"Total files processed: {len(metadata['processed_files'])}")