from PIL import Image
import pytesseract
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator

import os

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