# database.py
# Save and load customer segments from MongoDB Atlas

import pymongo
import os
from dotenv import load_dotenv
import pandas as pd
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mongo_client():
    """Connect to MongoDB Atlas"""
    try:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI not found in .env")
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info("✅ Connected to MongoDB Atlas!")
        return client
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return None


def save_rfm_to_mongodb(rfm_df):
    """Save RFM + Cluster data to MongoDB"""
    client = get_mongo_client()
    if not client:
        return False

    try:
        db_name = os.getenv("DB_NAME", "customer_segments_db")
        collection_name = os.getenv("COLLECTION_NAME", "customers_rfm_segments")
        db = client[db_name]
        collection = db[collection_name]

        records = rfm_df.reset_index()
        records['CustomerID'] = records['CustomerID'].astype(int)
        records['Recency'] = records['Recency'].astype(int)
        records['Frequency'] = records['Frequency'].astype(int)
        records['Monetary'] = records['Monetary'].astype(float)
        records['Cluster'] = records['Cluster'].astype(int)
        records['created_at'] = pd.Timestamp.utcnow()

        for record in records.to_dict('records'):
            collection.update_one(
                {"CustomerID": record["CustomerID"]},
                {"$set": record},
                upsert=True
            )

        logger.info(f"✅ Saved {len(records)} customers to MongoDB.")
        return True

    except Exception as e:
        logger.error(f"❌ Save failed: {e}")
        return False

    finally:
        client.close()


def load_rfm_from_mongodb():
    """Load segmented data from MongoDB"""
    client = get_mongo_client()
    if not client:
        return None

    try:
        db_name = os.getenv("DB_NAME", "customer_segments_db")
        collection_name = os.getenv("COLLECTION_NAME", "customers_rfm_segments")
        collection = client[db_name][collection_name]

        data = list(collection.find({}))
        if not data:
            logger.warning("⚠️ No data found in MongoDB.")
            return None

        df = pd.DataFrame(data)
        if 'CustomerID' not in df.columns:
            return None

        df.set_index('CustomerID', inplace=True)
        df = df[['Recency', 'Frequency', 'Monetary', 'Cluster']]
        logger.info(f"✅ Loaded {len(df)} customers from MongoDB.")
        return df

    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
        return None

    finally:
        client.close()


def test_connection():
    """Test MongoDB connection only"""
    client = get_mongo_client()
    if client:
        print("🎉 MongoDB connection successful!")
        client.close()
        return True
    else:
        print("❌ MongoDB connection failed.")
        return False