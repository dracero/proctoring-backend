from PIL import Image
from io import BytesIO
import torch 
import base64
from transformers import pipeline
import db
from project_utils import Logger, ErrorHandler  # Import the Logger and ErrorHandler

# Initialize the logger for this module
logger = Logger(__name__)

class Models:
    nlp = None
    image_processor = None
    object_detection_model = None

def get_screenshot_report(student_email: str, student_test: str,student_report: dict):
    logger.log('Fetching screenshot from the database...')
    screenshots = db.get_mongo_collection("screenshot", student_email)
    if not screenshots:
        student_report["screenshot"] = "FAIL: No screenshot."
        return
    # Only keep the latest screenshot
    screenshot_data = screenshots[-1]
    base64_image = screenshot_data["image"]
    logger.log('Processing image data...')
    # Convert base64 to JPG
    image_data = base64.b64decode(base64_image.split(',')[1])
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')
    logger.log('Running image through the pipeline...')
    # Use the document-question-answering pipeline
    title_data = Models.nlp(image, "What does the title say?")[0]
    score = title_data.get('score', 0)
    answer = title_data.get('answer', '').lower()
    if score < 0.5:
        result = False
    else:
        result = answer == student_test.lower()
    logger.log('Appending to the report...')
    student_report["screenshot"] = "SUCCESS" if result else "FAIL: Wrong text."

def get_OOF_report(student_email: str, student_report: dict):
    logger.log(f"Checking out of frame data for student: {student_email}")
    try:
        # Query the outOfFrame collection for documents with the student's email
        out_of_frame_data = db.get_mongo_collection("outOfFrame", student_email)
        # If any documents are found, append a new field to the student_report
        if out_of_frame_data:
            student_report["outOfFrame"] = "FAIL"
            logger.log(f"Student {student_email} has failed the out of frame test.")
        else:
            student_report["outOfFrame"] = "SUCCESS"
            logger.log(f"Student {student_email} has passed the out of frame test.")
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)

def get_blur_report(student_email: str, student_report: dict):
    logger.log(f"Checking blur data for the student: {student_email}")
    try:
        # Query the outOfFrame collection for documents with the student's email
        out_of_frame_data = db.get_mongo_collection("blur", student_email)
        # If any documents are found, append a new field to the student_report
        if out_of_frame_data:
            student_report["blur"] = "FAIL"
            logger.log(f"Student {student_email} has failed the blur test.")
        else:
            student_report["blur"] = "SUCCESS"
            logger.log(f"Student {student_email} has passed the blur test.")
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)        

def get_OD_report(student_email: str, student_report: dict):
    logger.log(f"Running object detection for student: {student_email}")
    try:
        # Query the "periodicPhotos" collection and retrieve all images
        periodic_photos_data = db.get_mongo_collection("periodicPhotos", student_email)
        
        if not periodic_photos_data:
            student_report["objectDetection"] = "FAIL: No periodic photos found."
            return
        
        for photo_data in periodic_photos_data:
            # Convert base64 to Image
            base64_image = photo_data["image"]
            image_data = base64.b64decode(base64_image.split(',')[1])
            image = Image.open(BytesIO(image_data))    
            image = image.convert('RGB')        
            # Run the images through the Object Detection model
            inputs = Models.image_processor(images=image, return_tensors="pt")
            outputs = Models.object_detection_model(**inputs)        
            # Convert outputs to COCO API
            target_sizes = torch.tensor([image.size[::-1]])
            results = Models.image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]            
            # Determine whether the function should return "SUCCESS" or "FAIL"
            person_detected = False
            cell_phone_detected = False
            cell_phone_confidence = 0
            
            for score, label in zip(results["scores"], results["labels"]):
                detected_object = Models.object_detection_model.config.id2label[label.item()]
                if detected_object == "person":
                    person_detected = True
                elif detected_object == "cell phone":
                    cell_phone_detected = True
                    cell_phone_confidence = score.item()
            
            if not person_detected or (cell_phone_detected and cell_phone_confidence > 0.9):
                student_report["objectDetection"] = "FAIL: Unauthorized object detected or person not detected."
                return
        
        student_report["objectDetection"] = "SUCCESS"
        logger.log(f"Student {student_email} has passed the object detection test.")
        
        student_report["objectDetection"] = "SUCCESS"
        logger.log(f"Object detection completed successfully for student: {student_email}")
        
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e))
