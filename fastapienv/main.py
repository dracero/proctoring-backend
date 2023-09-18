from fastapi import FastAPI
from pymongo import MongoClient
import os
# import database utilities
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
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


def get_student_report(student_email: str):
    logger.log(f"Generating report for student: {student_email}", logging.INFO)
    # Initialize an empty dictionary that will hold the report for the student
    student_report = dict()
    student_report["student"] = student_email
    try:
        # Process student screenshot and append results to the report
        ML.get_screenshot_report(student_email, student_report)
        # ... other report functions will go here ...
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)
    
    logger.log(f"Report generated successfully for student: {student_email}", logging.INFO)
    return student_report


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
        report = get_student_report(student_email)
        # Append the report to the list of reports
        student_reports.append(report)    
    logger.log(f"Reports generated successfully for test: {test_name}", logging.INFO)
    return student_reports

# BDD con iframe + lista de alumnos de ese exámen  
# Sería algo así:
# @app.get(/report/{test_name})
# async def get_report_for_test(test_name:str)
#   # look up all students that performed this test in the 'tests' database (return the list of emails)
#   # create empty list of dictionaries student_reports
#   # call get_student_report for each student of the list and append the report to the list
#   # return the student reports for this test









