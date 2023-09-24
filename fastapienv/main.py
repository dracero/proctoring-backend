from fastapi import FastAPI
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
        ML.Models.nlp = pipeline("document-question-answering", model="impira/layoutlm-document-qa",)
        ML.Models.image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        ML.Models.object_detection_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
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
        ML.get_OOF_report(student_email,student_report) 
        ML.get_blur_report(student_email,student_report) 
        ML.get_OD_report(student_email,student_report)
        # ... other report functions will go here ...
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)
    
    logger.log(f"Report generated successfully for student: {student_email}", logging.INFO)
    return student_report

# Consultas:
    ################################
    #   El checkeo de ML_screenshot se encuentra harcodeado,
    #   ¿Cómo manejamos los distintos checkeos para cada exámen?
    #   IDEA 1: Crear una 'screenshot_checklist' según el nombre de el exámen
    #   IDEA 2: Si solo verificamos el título de el exámen, entonces podemos simplemente
    #   pasarlo como argumento y comparar con él (ESTA)
    ################################
    #   ¿Cómo manejamos los distintos tipos de fallos?
    #   IDEA 1: Puedo dejarlo como esta y que el front se ocupe de el manejo.
    ################################
    #   ¿Qué módulo de ML implemento para la semana que viene?
    #   EXPLICACIÓN: 
    #   Pasar los Screenshots Periodicos por el detector de objetos

