from PIL import Image
from io import BytesIO
import base64
from transformers import pipeline
import db
from project_utils import Logger, ErrorHandler  # Import the Logger and ErrorHandler

# Initialize the logger for this module
logger = Logger(__name__)

class Models:
    nlp = None

def get_screenshot_report(student_email: str, student_report: dict):
    logger.log('Fetching screenshot from the database...')
    screenshots = db.get_mongo_collection("screenshot", student_email)
    if not screenshots:
        student_report["screenshot"] = "No screenshots found for the given student email."
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
        result = answer == "formulario de prueba"
    logger.log('Appending to the report...')
    student_report["screenshot"] = "success" if result else "fail"
