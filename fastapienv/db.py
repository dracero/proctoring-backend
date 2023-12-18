# Module for database communication

from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import json_util
import json
from project_utils import Logger, ErrorHandler  # Import the Logger and ErrorHandler

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)
db = client["proctoring"]

VALID_COLLECTIONS = ["blur", "conversations", "firstPhoto", "screenshot","test","periodicPhotos","outOfFrame","reports","ObjectDetectionData"]

def get_mongo_collection(collection_name: str, query: dict = None):
    # Ensure the collection name is valid
    if collection_name not in VALID_COLLECTIONS:
        raise ErrorHandler.InvalidCollectionError(f"Invalid collection name: {collection_name}")
    # Fetch the collection based on the provided name
    collection = db[collection_name]
    # If a query dictionary was passed, use it to filter the documents
    documents = list(collection.find(query or {}))

    return json.loads(json_util.dumps(documents))

def insert_into_mongo_collection(collection_name: str, data: dict):
    # Ensure the collection name is valid
    if collection_name not in VALID_COLLECTIONS:
        raise ErrorHandler.InvalidCollectionError(f"Invalid collection name: {collection_name}")
    try:
        existing_data = get_mongo_collection(collection_name, data)

        if existing_data:
            return  # If the data already exists, simply return without inserting
        collection = db[collection_name]
        collection.insert_one(data)
    except Exception as e:
        raise ErrorHandler.DatabaseConnectionError(f"Error inserting data into {collection_name} collection: {str(e)}")
    
def clear_mongo_collection(collection_name: str):
    # Ensure the collection name is valid
    if collection_name not in VALID_COLLECTIONS:
        raise ErrorHandler.InvalidCollectionError(f"Invalid collection name: {collection_name}")
    try:
        collection = db[collection_name]
        collection.delete_many({})
    except Exception as e:
        raise ErrorHandler.DatabaseConnectionError(f"Error clearing {collection_name} collection: {str(e)}")    