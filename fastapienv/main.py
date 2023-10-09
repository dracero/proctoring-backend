from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import os
# import database utilities
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
import json
# import machine learning models
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
from io import BytesIO
import base64
# import project utilities
from datetime import datetime, timedelta
from dateutil import parser
import logging 
import db
import os
import ml_utils as ML 
from project_utils import Logger, ErrorHandler  # <-- Importing Logger and ErrorHandler

app = FastAPI()

# Initialize the logger
logger = Logger(name="main_module")

# Initialize all machine learning models at startup
@app.on_event("startup")
async def load_models():
    logger.log("Initializing machine learning models...", logging.INFO)
    hf_token = os.getenv("HF_TOKEN")
    try:
        ML.Models.nlp = pipeline("document-question-answering", model="impira/layoutlm-document-qa",)
        ML.Models.image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        ML.Models.object_detection_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
        ML.Models.speech_classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli",token=hf_token)  # Initializing the new model at startup
        logger.log("Machine learning models initialized successfully.", logging.INFO)
    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(f"Error initializing models: {str(e)}", logging.ERROR)

@app.get("/reports/{test_name}")
async def get_reports_for_test(test_name: str):
    logger.log(f"Generating reports for test: {test_name}", logging.INFO)    
    # Retrieve the list of students who took the specified test
    students_list = retrieve_students(test_name)    
    logger.log(f"List of students who took the specified test: {students_list}", logging.INFO)
    # Initialize an empty list to hold the reports for each student
    student_reports = []    
    for student_email in students_list:
        # Generate the report for the student
        report = get_student_report(student_email,test_name)
        # Append the report to the list of reports
        student_reports.append(report)    
    logger.log(f"Reports generated successfully for test: {test_name}", logging.INFO)
    return student_reports

@app.get("/reports/{test_name}/{student_email}/screenshot")
async def get_screenshot_details(test_name: str, student_email: str):
    try:
        # Constructing the query to find the document that matches the given test_name and student_email
        query = {"exam": test_name, "student": student_email}
        # Retrieving the entry from the screenshot collection in the database
        screenshot_entries = db.get_mongo_collection("screenshot", query)        
        # If there are no entries that match the query, return the status message
        if not screenshot_entries:
            return "The student sent no screenshot"
        # Assuming there is only one entry that matches the query, return the image field of the entry
        return screenshot_entries[0]["image"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/reports/{test_name}/{student_email}/out_of_frame")
async def get_out_of_frame_details(test_name: str, student_email: str):
    try:
        out_of_frame_entries = db.get_mongo_collection("outOfFrame", {"exam": test_name, "student": student_email})
        final_data = []
        
        # Retrieving all corresponding images from the periodicPhotos collection in the database for the given test_name and student_email
        all_periodic_photos_entries = db.get_mongo_collection("periodicPhotos", {"exam": test_name, "student": student_email})
        
        for entry in out_of_frame_entries:
            time_str = entry.get("time")
            duration = entry.get("duration")
            time_dt = parser.parse(time_str)
            end_time_dt = time_dt + timedelta(seconds=duration)
            
            # Filtering the periodic_photos_entries based on the time frame
            filtered_periodic_photos_entries = [
                photo_entry for photo_entry in all_periodic_photos_entries
                if time_dt <= parser.parse(photo_entry.get("time")) <= end_time_dt
            ]
            
            # Extracting the base64 images from the filtered entries
            images_base64 = [photo_entry.get("image") for photo_entry in filtered_periodic_photos_entries]
            
            final_data.append({
                "time": time_str,
                "duration": duration,
                "images": images_base64
            })
        
        return final_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/reports/{test_name}/{student_email}/blur")
async def get_blur_details(test_name: str, student_email: str):
    try:
        # Constructing the query to find the documents that match the given test_name and student_email in blur collection
        query = {
            "student": student_email,
            "exam": test_name
        }
        logger.log(f"Connecting to blur database for student: {student_email}", logging.INFO)
        # Retrieving the corresponding entries from the blur collection in the database
        blur_entries = db.get_mongo_collection("blur", query)
        logger.log(f"Retrieving blur entries: {blur_entries}", logging.INFO)
        # Extracting the time and message from each entry and returning them
        return [{"time": entry["time"], "msg": entry["msg"]} for entry in blur_entries]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/reports/{test_name}/{student_email}/object_detection")
async def get_object_detection_details(test_name: str, student_email: str):
    try:
        # Constructing the query to find the documents that match the given test_name and student_email in blur collection
        query = {
            "student": student_email,
            "exam": test_name
        }
        logger.log(f"Connecting to periodicPhotos database for student: {student_email}", logging.INFO)
        # Retrieving the corresponding entries from the blur collection in the database
        photo_entries = db.get_mongo_collection("periodicPhotos", query)
        # Extracting the time and message from each entry and returning them
        return [{"time": entry["time"], "image": entry["image"]} for entry in photo_entries]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/reports/{test_name}/{student_email}/speech_detection")
async def get_speech_detection_details(test_name: str, student_email: str):
    try:
        # Constructing the query to find the documents that match the given test_name and student_email in blur collection
        query = {
            "student": student_email,
            "exam": test_name
        }
        logger.log(f"Connecting to speech database for student: {student_email}", logging.INFO)
        # Retrieving the corresponding entries from the blur collection in the database
        speech_entries = db.get_mongo_collection("conversations", query)
        # Extracting the time and message from each entry and returning them
        return [{"time": entry["time"], "conversation": entry["conversation"]} for entry in speech_entries]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def retrieve_students(exam_name: str):
    try:
        logger.log(f"Fetching the list of students for the test: {exam_name}", logging.INFO)
        students_data = db.get_mongo_collection("test")
        students_list = [data["student"] for data in students_data if data["exam"].lower() == exam_name.lower()]
        logger.log(f"Students list: {students_list}", logging.INFO)
        return students_list
    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)
        return []


def get_student_report(student_email: str,student_test : str):
    logger.log(f"Generating report for student: {student_email}", logging.INFO)
    # Initialize an empty dictionary that will hold the report for the student
    student_report = dict()
    student_report["student"] = student_email
    student_report["test"] = student_test 
    try:
        # Process student screenshot and append results to the report
        ML.get_screenshot_report(student_email,student_test,student_report)
        ML.get_OOF_report(student_email,student_test,student_report) 
        ML.get_blur_report(student_email,student_test,student_report) 
        ML.get_OD_report(student_email,student_test,student_report)
        ML.get_speech_report(student_email,student_test,student_report)
        # ... other report functions will go here ...
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)
    
    logger.log(f"Report generated successfully for student: {student_email}", logging.INFO)
    return student_report

