from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from langdetect import detect
import pytesseract
import preprocess_image as ppImg
import translate_image as trnsImg

import easyocr


# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

if __name__ == "__main__":

    img_path = '../test/images/tamil4.jpg'
    image = Image.open(img_path)

    #transform iamge
    preprocessor = ppImg.PreprocessImage(metrics=['grayscale','remove_noise'])
    image_np = preprocessor.transform_image(image)

    #extract text
    converted_image = Image.fromarray(image_np)

    #detect language
    script_name, _ = trnsImg.detect_language(img_path)

    print(script_name)

    if script_name == "Han":
        text = pytesseract.image_to_string(converted_image, lang='chi_sim')
    elif script_name == "Tamil":
        text = pytesseract.image_to_string(converted_image, lang='tam')
    elif script_name == "Arabic":
        text1 = pytesseract.image_to_string(converted_image, lang='chi_sim')
        text2 = pytesseract.image_to_string(converted_image, lang='tam')
        if len(text1) < len(text2):
            text = text1
        else:
            text = text2
    else:
        text = pytesseract.image_to_string(converted_image)
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)
