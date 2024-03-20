from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from langdetect import detect
import pytesseract
import preprocess_image as ppImg
from googletrans import Translator


# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

if __name__ == "__main__":

    img_path = "../test/images/8b52fi.png"
    image = Image.open(img_path)


    #transform iamge
    preprocessor = ppImg.PreprocessImage(metrics=['grayscale','remove_noise','erode'])
    image_np = preprocessor.transform_image(image)

    #extract text
    converted_image = Image.fromarray(image_np)
    text = pytesseract.image_to_string(converted_image, lang='chi_sim')
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)

    #translate text
    translator = Translator()
    translate_text = translator.translate(text) 

    print(translate_text.text)