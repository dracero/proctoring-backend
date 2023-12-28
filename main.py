from fastapi import FastAPI, BackgroundTasks, HTTPException
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

# Startup Events
 # Load all Machine Learning Models
 # Refresh Reports (In case new data has beene added to the database)
@app.on_event("startup")
async def startup():
    # Start events on the Background
    init_background_tasks(BackgroundTasks())

def init_background_tasks(background_tasks: BackgroundTasks):
    background_tasks.add_task(load_models)
    background_tasks.add_task(refresh_reports)

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

async def refresh_reports():
    try:
        logger.log("Initializing report refresh", logging.INFO)
        # Step 1: Query all exam names from the "test" collection
        all_tests_data = db.get_mongo_collection("test")
        all_exam_names = set([data["exam"] for data in all_tests_data])
        # Step 2: Query existing reports from the database
        all_reports_data = db.get_mongo_collection("reports")
        reported_exam_names = set([data["test"] for data in all_reports_data])
        # Step 3: Check if new exams have been aded
        missing_reports = all_exam_names - reported_exam_names
        # Step 4: Generate reports for those exams
        for exam_name in missing_reports:
            await produce_report(exam_name)
            logger.log(f"Report generated and stored for exam: {exam_name}", logging.INFO)

    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(f"Error refreshing reports: {str(e)}", logging.ERROR)


# Get The specific report for a specific test
@app.get("/reports/{test_name}")
async def get_reports_for_test(test_name: str):
    try:
        # Step 1: Query the entry of the "reports" collection that corresponds to this test name
        query = {"test": test_name}
        report_data = db.get_mongo_collection("reports", query)

        # Step 2: Check if the report data exists for the given test name
        if not report_data:
            raise HTTPException(status_code=404, detail=f"No reports found for test: {test_name}")

        # Step 3: Return the 'reports' field of this entry
        return report_data[0]["reports"]

    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(f"Error fetching reports for test {test_name}: {str(e)}", logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Error fetching reports for test: {test_name}")

# Partial Refresh will be called from the Front-End on it's startup
@app.get("/reports/refresh/partial")
async def partial_refresh_reports():
    try:
        logger.log("Initializing report refresh", logging.INFO)
        # Step 1: Query all exam names from the "test" collection
        all_tests_data = db.get_mongo_collection("test")
        all_exam_names = set([data["exam"] for data in all_tests_data])
        # Step 2: Query existing reports from the database
        all_reports_data = db.get_mongo_collection("reports")
        reported_exam_names = set([data["test"] for data in all_reports_data])
        # Step 3: Check if new exams have been aded
        missing_reports = all_exam_names - reported_exam_names
        # Step 4: Generate reports for those exams
        for exam_name in missing_reports:
            await produce_report(exam_name)
            logger.log(f"Report generated and stored for exam: {exam_name}", logging.INFO)

    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(f"Error refreshing reports: {str(e)}", logging.ERROR)

# Full Refresh to be used if ever needed
@app.get("/reports/refresh/full")
async def full_refresh_reports():
    try:
        # Step 1: Delete all elements from the reports collection
        db.clear_mongo_collection("reports")
        logger.log("Cleared all reports from the database.", logging.INFO)
        # Step 2: Query all exam names from the test collection
        test_data = db.get_mongo_collection("test")
        exam_names = set([data["exam"] for data in test_data])
        # Step 3: Call produce_report for each exam name
        for exam_name in exam_names:
            await produce_report(exam_name)
            logger.log(f"Refreshed report for test: {exam_name}", logging.INFO)
        return {"status": "success", "message": "Reports fully refreshed."}

    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(f"Error during full refresh: {str(e)}", logging.ERROR)
        raise HTTPException(status_code=500, detail="Error during full refresh of reports.")

# Get exams. To be used by the Front-End to list the available examinations    
@app.get("/exams")
async def get_exam_names():
    try:
        # Step 1: Query all exam names from the test collection
        test_data = db.get_mongo_collection("test")
        exam_names = set([data["exam"] for data in test_data])
        # Step 3: Call produce_report for each exam name
        return exam_names

    except Exception as e:
        ErrorHandler.handle_exception(e)
        logger.log(f"Error loading exam names: {str(e)}", logging.ERROR)
        raise HTTPException(status_code=500, detail="Error during exam names load.")


# Get a specific screenshot report
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

# Get a specific out of frame report
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

# Get a specific blur report
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

# Get a specific object detection report
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
        photo_entries = db.get_mongo_collection("ObjectDetectionData", query)
        # Extracting the time and message from each entry and returning them
        return [{"time": entry["time"], "image": entry["image"]} for entry in photo_entries]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Get a specific speech detection report
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

# Produce full report for a given exam
async def produce_report(test_name: str):
    logger.log(f"Generating reports for test: {test_name}", logging.INFO)    
    # Step 1: Retrieve the list of students who took the test
    students_list = retrieve_students(test_name)    
    logger.log(f"List of students who took the specified test: {students_list}", logging.INFO)
    # Step 2: Initialize an empty list to hold the reports for each student
    student_reports = [] 
    # Step 3: Generate report for each student for thist est   
    for student_email in students_list:
        # Generate the report for the student
        report = get_student_report(student_email,test_name)
        # Append the report to the list of reports
        student_reports.append(report)    
    # Step 4: Create the final report structure
    final_report = {
        "test": test_name,
        "reports": student_reports
    }
    # Step 5: Store the final report in the database
    db.insert_into_mongo_collection("reports",final_report)
    logger.log(f"Reports generated successfully for test: {test_name}", logging.INFO)


def get_student_report(student_email: str,student_test : str):
    logger.log(f"Generating report for student: {student_email}", logging.INFO)
    # Initialize an empty dictionary that will hold the report for the student
    student_report = dict()
    student_report["student"] = student_email
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

