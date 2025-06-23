"""
Ingest wind turbine CSV data and store in MongoDB.
"""
import pandas as pd
from pymongo import MongoClient

def ingest_to_mongodb(csv_path, mongo_uri="mongodb://localhost:27017/", db_name="wind_monitoring"):
    df = pd.read_csv(csv_path)
    client = MongoClient(mongo_uri)
    db = client[db_name]
    db["raw_data"].delete_many({})
    db["raw_data"].insert_many(df.to_dict(orient="records"))
    print(f"Ingested {len(df)} records to MongoDB '{db_name}.raw_data'.")

if __name__ == "__main__":
    ingest_to_mongodb(r"C:\Users\DELL\OneDrive\Desktop\wind_turbine\wind_turbine_mlops\data\wind_turbine_Data.csv")

#save requirements.txt
