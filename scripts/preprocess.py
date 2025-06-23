# """
# Preprocess raw data: load from MongoDB, handle nulls, encode categorical, scale numerics, and save cleaned data back to MongoDB.
# """
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from pymongo import MongoClient

# def load_from_mongodb(uri="mongodb://localhost:27017/", db_name="wind_monitoring", collection_name="raw_data"):
#     client = MongoClient(uri)
#     db = client[db_name]
#     data = list(db[collection_name].find())
#     if data and '_id' in data[0]:
#         for d in data:
#             d.pop('_id', None)
#     return pd.DataFrame(data)

# def save_to_mongodb(df, uri="mongodb://localhost:27017/", db_name="wind_monitoring", collection_name="cleaned_data"):
#     client = MongoClient(uri)
#     db = client[db_name]
#     db[collection_name].delete_many({})
#     db[collection_name].insert_many(df.to_dict(orient="records"))
#     print(f"Saved {len(df)} cleaned records to MongoDB '{db_name}.{collection_name}'.")

# def preprocess(df):
#     df = df.dropna()
#     if 'TurbineName' in df.columns:
#         df['TurbineName'] = df['TurbineName'].astype('category').cat.codes
#     features = df.drop(columns=['Power'])
#     target = df['Power']
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
#     # Save cleaned data (features + target) to MongoDB
#     cleaned_df = features.copy()
#     cleaned_df['Power'] = target
#     save_to_mongodb(cleaned_df)
#     return features_scaled, target, scaler

# if __name__ == "__main__":
#     df = load_from_mongodb()
#     X, y, scaler = preprocess(df)
#     print("Preprocessing complete. Feature shape:", X.shape)
#     print("First 5 features:\n", X[:5])


"""
Preprocess raw data: load from MongoDB, handle nulls, encode categorical,
scale numerics, save cleaned data and scaler for reuse.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import joblib
import os

def load_from_mongodb(uri="mongodb://localhost:27017/", db_name="wind_monitoring", collection_name="raw_data"):
    client = MongoClient(uri)
    db = client[db_name]
    data = list(db[collection_name].find())
    if data and '_id' in data[0]:
        for d in data:
            d.pop('_id', None)
    return pd.DataFrame(data)

def save_to_mongodb(df, uri="mongodb://localhost:27017/", db_name="wind_monitoring", collection_name="cleaned_data"):
    client = MongoClient(uri)
    db = client[db_name]
    db[collection_name].delete_many({})
    db[collection_name].insert_many(df.to_dict(orient="records"))
    print(f"Saved {len(df)} cleaned records to MongoDB '{db_name}.{collection_name}'.")

def save_scaler(scaler, path="models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

def preprocess(df):
    df = df.dropna()
    if 'TurbineName' in df.columns:
        df['TurbineName'] = df['TurbineName'].astype('category').cat.codes

    features = df.drop(columns=['Power'])
    target = df['Power']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Save scaler
    save_scaler(scaler)

    # Save clean but unscaled data to MongoDB
    cleaned_df = features.copy()
    cleaned_df['Power'] = target
    save_to_mongodb(cleaned_df)

    return features_scaled, target

if __name__ == "__main__":
    df = load_from_mongodb()
    X_scaled, y = preprocess(df)
    print("Preprocessing complete. Feature shape:", X_scaled.shape)
    print("First 5 features:\n", X_scaled[:5])
    print("Target values:\n", y.head())