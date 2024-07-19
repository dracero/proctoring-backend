from PIL import Image
from io import BytesIO
import torch 
import base64
from transformers import pipeline
import db
from project_utils import Logger, ErrorHandler  # Import the Logger and ErrorHandler

# Initialize the logger for this module
logger = Logger(__name__)

"""
# Wrapper for ML models. The attributes are initialized at startup. 
class Models:
    nlp = None
    image_processor = None
    object_detection_model = None
    speech_classifier = None

def get_screenshot_report(student_email: str, student_test: str, student_report: dict):
    logger.log(f"Generating screenshot report for student: {student_email} for the test: {student_test}")
    screenshots = db.get_mongo_collection("screenshot", {"student": student_email, "exam": student_test})
    if not screenshots:
        student_report["screenshot"] = "FAIL: No screenshot."
        return
    # Only keep the latest screenshot
    screenshot_data = screenshots[-1]
    base64_image = screenshot_data["image"]
    # Convert base64 to JPG
    image_data = base64.b64decode(base64_image.split(',')[1])
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')
    try:
        # Use the document-question-answering pipeline
        title_data = Models.nlp(image, "What does the title say?")[0]
        score = title_data.get('score', 0)
        answer = title_data.get('answer', '').lower()
        if score < 0.5:
            result = False
        else:
            result = answer == student_test.lower()
        student_report["screenshot"] = "SUCCESS" if result else "FAIL: Wrong text."
    except Exception as e:
        # If there's an error in reading the text, log the exception and set the report to indicate failure due to no text.
        logger.log(f"Error processing screenshot for student: {student_email} for the test: {student_test}. Error: {e}")
        student_report["screenshot"] = "FAIL: No text."



def get_OOF_report(student_email: str, student_test: str, student_report: dict):
    logger.log(f"Generating out of frame report for student: {student_email} for the test: {student_test}")
    try:
        # Query the outOfFrame collection for documents with the student's email and test
        out_of_frame_data = db.get_mongo_collection("outOfFrame", {"student": student_email, "exam": student_test})
        if out_of_frame_data:
            student_report["outOfFrame"] = "FAIL"
            logger.log(f"Student {student_email} has failed the out of frame test for {student_test}.")
        else:
            student_report["outOfFrame"] = "SUCCESS"
            logger.log(f"Student {student_email} has passed the out of frame test for {student_test}.")
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)

def get_blur_report(student_email: str, student_test: str, student_report: dict):
    logger.log(f"Generating blur report for student: {student_email} for the test: {student_test}")
    try:
        # Query the blur collection for documents with the student's email and test
        blur_data = db.get_mongo_collection("blur", {"student": student_email, "exam": student_test})
        if blur_data:
            student_report["blur"] = "FAIL"
        else:
            student_report["blur"] = "SUCCESS"
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)

def get_OD_report(student_email: str, student_test: str, student_report: dict):
    logger.log(f"Running object detection for student: {student_email} for test: {student_test}")
    try:
        # Query the "periodicPhotos" collection and retrieve all images for the student's email and test
        periodic_photos_data = db.get_mongo_collection("periodicPhotos", {"student": student_email, "exam": student_test})
        if not periodic_photos_data:
            student_report["objectDetection"] = "FAIL: No periodic photos found."
            return

        object_detected = False  # Flag to track if any object is detected

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

            person_detected = False
            cell_phone_detected = False
            cell_phone_confidence = 0

            for score, label in zip(results["scores"], results["labels"]):
                detected_object = Models.object_detection_model.config.id2label[label.item()]
                print('detected_object:', detected_object)
                if detected_object == "person":
                    person_detected = True
                elif detected_object == "cell phone":
                    cell_phone_detected = True
                    cell_phone_confidence = score.item()

            # If an object is detected, store the photo data in ObjectDetectionData collection
            if cell_phone_detected and cell_phone_confidence > 0.9:
                object_detected = True
                detected_data = {
                    "student": photo_data["student"],
                    "exam": photo_data["exam"],
                    "time": photo_data["time"],
                    "image": photo_data["image"]
                }
                db.insert_into_mongo_collection("ObjectDetectionData", detected_data)

        # Set the report status based on whether any object was detected
        student_report["objectDetection"] = "FAIL" if object_detected else "SUCCESS"
                
    except ErrorHandler.Error as e:
        ErrorHandler.handle_exception(e)
        logger.log(str(e), logging.ERROR)


def get_speech_report(student_email: str, student_test: str, student_report: dict):
    logger.log(f"Running speech recognition for student: {student_email} for test: {student_test}")
    try:
        # Initialize the zero-shot-classification pipeline with the specified model
        classifier = Models.speech_classifier        
        # Query the 'conversations' collection for the specified student and test
        conversations_data = db.get_mongo_collection("conversations", {"student": student_email, "exam": student_test})     
        # If no conversations are found, we can append SUCCESS and return early
        if not conversations_data:
            student_report["speech"] = "SUCCESS: No conversations found."
            return        
        # Query the 'test' collection to get the themes for the specified student and test
        test_data = db.get_mongo_collection("test", {"student": student_email, "exam": student_test})
        # Extract the themes and split them into a list of labels
        themes = test_data[0].get("themes", "").split(',') if test_data else []
        # Initialize a variable to hold the merged transcript
        merged_transcript = ""        
        # Process each conversation snippet individually
        for conversation in conversations_data:
            snippet = conversation.get("conversation", "")
            merged_transcript += snippet + " "  # Append the snippet to the merged transcript            
            # Classify the snippet and check the probabilities
            result = classifier(snippet, themes)
            for label, score in zip(result["labels"], result["scores"]):
                if score > 0.5:
                    student_report["speech"] = f"FAIL: Detected theme '{label}' in conversation snippet with probability {score}."
                    return  # Return early as we have detected a theme        
        # Process the merged transcript
        result = classifier(merged_transcript, themes)
        for label, score in zip(result["labels"], result["scores"]):
            if score > 0.5:
                student_report["speech"] = f"FAIL: Detected theme '{label}' in merged transcript with probability {score}."
                return  # Return early as we have detected a theme        
        # If no themes are detected in both cases, append SUCCESS to the student_report
        student_report["speech"] = "SUCCESS: No themes detected."
        
    except Exception as e:
        # Handle any exceptions that occur during the process
        student_report["speech"] = f"ERROR: An error occurred while processing speech report: {str(e)}"
"""