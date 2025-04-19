# Engagement Prediction Model

This project trains a model to predict user engagement based on data stored in an R2 bucket.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure environment variables:**
    Create a `.env` file in the root directory with your R2 credentials:
    ```
    R2_ACCESS_KEY=your_access_key
    R2_SECRET_KEY=your_secret_key
    R2_ENDPOINT=your_endpoint_url
    R2_BUCKET=your_bucket_name
    R2_PREFIX=your_data_prefix  # Optional: If your data is in a specific folder
    ```

## Training

To train the model (or update it with new data), run:

```bash
python train_full_from_r2.py
```

The script will:
- Load metadata about previously processed files.
- Identify and process only new data files from the R2 bucket.
- Load the existing model (if found) or create a new one.
- Train/update the model using the new data.
- Save the updated model and metadata. 

## Prediction

A sample prediction script `predict.py` is provided to demonstrate how to load the trained model and make predictions.

It currently uses a hardcoded example data point. You can modify this script to take dynamic input or integrate it into your application.

To run the sample prediction:

```bash
python predict.py
```

This will output the predicted engagement label (0 for no engagement, 1 for engagement) and the corresponding probability for the sample data. 