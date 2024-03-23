import pytesseract
import re
import cv2 as cv
import os
from PIL import Image

def load_images(directory):
    img_paths_list = []

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        img_paths_list.append(file_path)
    return img_paths_list

def convert_text(text):

    pass


def detect_language(img: Image):
    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0 -c min_characters_to_try=5")
        script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
        conf = re.search("Script confidence: (\d+\.?(\d+)?)", osd).group(1)
        print(script, conf)
        return script, float(conf)
    except Exception as e: 
        print (e)
        print("ERROR")
        return "null", 0.0

