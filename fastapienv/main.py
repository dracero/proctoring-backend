from fastapi import FastAPI
from pymongo import MongoClient
import os
# import database utilities
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from bson import json_util
import json
# import machine learning models
from transformers import pipeline
from PIL import Image
from io import BytesIO
import base64
# import project utilities
import logging 
import db
import ml_utils as ML 
from project_utils import Logger, ErrorHandler  # <-- Importing Logger and ErrorHandler

app = FastAPI()

# Initialize the logger
logger = Logger(name="main_module")

# Initialize all machine learning models at startup
@app.on_event("startup")
async def load_models():
    logger.log("Initializing machine learning models...", logging.INFO)
    try:
        ML.Models.nlp = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
        )
        logger.log("Machine learning models initialized successfully.", logging.INFO)
    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(f"Error initializing models: {str(e)}", logging.ERROR)

# General Endpoint to fetch the collections.
@app.get("/{collection_name}")
async def get_collection(collection_name: str, student_email: str = None):
    logger.log(f"Fetching data from collection: {collection_name} for student: {student_email}", logging.INFO)
    data = db.get_mongo_collection(collection_name, student_email)
    logger.log(f"Data fetched successfully from collection: {collection_name}", logging.INFO)
    return data


# Endpoint to produce the student's report
@app.get("/report/{student_email}")
async def get_student_report(student_email: str):
    logger.log(f"Generating report for student: {student_email}", logging.INFO)
    # initialize empty dictionary that will hold the full report
    student_report = dict()
    # run reporting functions   
    try:
        # process student screenshot and append results to the report
        ML.get_screenshot_report(student_email, student_report)
        # ... other report functions will go here ...
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)     
    
    logger.log(f"Report generated successfully for student: {student_email}", logging.INFO)    
    return student_report