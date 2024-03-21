from PIL import Image
import pytesseract
import re
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator
import cv2 as cv

import os

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

def load_images(directory):
    img_paths_list = []

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        img_paths_list.append(file_path)
    return img_paths_list

def convert_text(text):

    translator = Translator()
    translate_text = translator.translate(text) 

    return translate_text


def detect_language(img_path):
    try:
        original_image = cv.imread(img_path)
        resized_image = cv.resize(original_image, (int(int(original_image.shape[1]) * 5), int(int(original_image.shape[1]) * 5)))
        osd = pytesseract.image_to_osd(resized_image)
        script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
        conf = re.search("Script confidence: (\d+\.?(\d+)?)", osd).group(1)
        return script, float(conf)
    except Exception as e: 
        print (e)
        return "Han", 0.0

