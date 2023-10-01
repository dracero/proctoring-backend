# Module for database communication

from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import json_util
import json

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)
db = client["proctoring"]

VALID_COLLECTIONS = ["blur", "conversations", "firstPhoto", "screenshot","test","periodicPhotos","outOfFrame"]

def get_mongo_collection(collection_name: str, query: dict = None):
    # Ensure the collection name is valid
    if collection_name not in VALID_COLLECTIONS:
        raise ValueError("Invalid collection name")
    # Fetch the collection based on the provided name
    collection = db[collection_name]
    # If a query dictionary was passed, use it to filter the documents
    documents = list(collection.find(query or {}))
    
    return json.loads(json_util.dumps(documents))