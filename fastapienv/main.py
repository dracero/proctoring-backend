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
import db

app = FastAPI()

@app.on_event("startup")
async def load_models():
    global nlp
    nlp = pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa",
    )

# General Endpoint to fetch the collections.
@app.get("/{collection_name}")
async def get_collection(collection_name: str, student_email: str = None):
    try:
        return db.get_mongo_collection(collection_name, student_email)
    except ValueError as e:
        return {"error": str(e)}

def get_screenshot_report(student_email: str, student_report: dict):
    print('Fetching screenshot from the database...')
    screenshots = db.get_mongo_collection("screenshot", student_email)
    if not screenshots:
        student_report["screenshot"] = "No screenshots found for the given student email."
        return
    
    # Only keep the latest screenshot
    screenshot_data = screenshots[-1]
    base64_image = screenshot_data["image"]
    print('Processing image data...')
    # Convert base64 to JPG
    image_data = base64.b64decode(base64_image.split(',')[1])
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')
    print('Running image through the pipeline...')
    # Use the document-question-answering pipeline
    title_data = nlp(image, "What does the title say?")[0]
    score = title_data.get('score', 0)
    answer = title_data.get('answer', '').lower()
    if score < 0.5:
        result = False
    else:
        result = answer == "formulario de prueba"
    print('Appending to the report...')
    student_report["screenshot"] = "success" if result else "fail"

# Endpoint to produce the student's report
@app.get("/report/{student_email}")
async def get_student_report(student_email: str):
    # initialize empty dictionary that will hold the full report
    student_report = dict()
    # process student screenshot and append results to the report
    get_screenshot_report(student_email, student_report)
    # ... other report functions will go here ...

    return student_report
