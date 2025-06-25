"""
Train regression model, evaluate, and log to MLflow.
Loads cleaned data from MongoDB (wind_monitoring.cleaned_data).
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
 

import mlflow
import mlflow.sklearn
try:
    from pymongo import MongoClient
except ImportError:
    raise ImportError("pymongo is not installed. Please install it with 'pip install pymongo'.")
from sklearn.preprocessing import StandardScaler
import joblib
def load_scaler(path="models/scaler.pkl"):
    return joblib.load(path)

def load_cleaned_data_from_mongodb(
    mongo_uri="mongodb://localhost:27017/",
    db_name="wind_monitoring",
    collection_name="cleaned_data"
):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    data = list(db[collection_name].find())
    if not data:
        raise ValueError("No cleaned data found in MongoDB.")
    df = pd.DataFrame(data)
    # Remove MongoDB's default '_id' field if present
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    return df

# def preprocess_for_training(df):
#     # Assume 'Power' is the target and all others are features
#     X = df.drop(columns=['Power'])
#     y = df['Power']
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return X_scaled, y, scaler
def preprocess_for_training(df):
    # Only use features, exclude the target variable 'Power'
    feature_cols = [col for col in df.columns if col != 'Power']
    X = df[feature_cols]
    y = df['Power']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ Save the correct scaler that was used
    joblib.dump(scaler, "models/scaler.pkl")
    print('Features used for training:', feature_cols)

    return X_scaled, y, scaler


def train_and_log_from_mongodb():
    df = load_cleaned_data_from_mongodb()
    X, y, scaler = preprocess_for_training(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    # X_train, y_train = X_train, y_train
    # only use 500 samples for training to speed up the process
    X_train, y_train = X_train[:100], y_train[:100]  # Use only 500 samples for faster training
    # Save the scaler for future inferen
    print('traing data first 5 rows:\n', X_train[:5])
    print('training data column names:\n', df.columns)
    print('y_test data first 5 records:',y_test[:5]) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 


    rmse = root_mean_squared_error(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # mlflow.set_tracking_uri("http://127.0.0.1:5000") # local MLflow server
    # For DagsHub, use the following tracking URI 
    
    mlflow.set_tracking_uri("https://dagshub.com/sksivakumar726/CICD_Wind.mlflow")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, artifact_path="model")
        # mlflow.sklearn.log_model(model, name="model")
        print(f"Run ID: {run.info.run_id} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.2f}")

if __name__ == "__main__":
    train_and_log_from_mongodb()