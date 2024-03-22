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

    img_path = '../test/images/racist.png'
    image = Image.open(img_path)

    # #transform iamge
    preprocessorBasic = ppImg.PreprocessImage(metrics=['grayscale','bilateral','thresholding'])
    image_np = preprocessorBasic.transform_image(image)

    preprocessorEng = ppImg.PreprocessImage(metrics=['grayscale','remove_noise','thresholding'])
    image_npEng = preprocessorEng.transform_image(image)

    preprocessorChi = ppImg.PreprocessImage(metrics=['grayscale','remove_noise'])
    image_npChi = preprocessorChi.transform_image(image)

    preprocessorTan = ppImg.PreprocessImage(metrics=['grayscale','thresholding'])
    image_npTam = preprocessorTan.transform_image(image)

    #extract text
    converted_image = Image.fromarray(image_np)
    converted_image.show()
    converted_imageEng = Image.fromarray(image_npEng)
    converted_imageEng.show()
    converted_imageChi = Image.fromarray(image_npChi)
    converted_imageChi.show()
    converted_imageTam = Image.fromarray(image_npTam)
    converted_imageTam.show()

    #detect language
    script_name, _ = trnsImg.detect_language(img_path)

    print(script_name)

    if script_name == "Han":
        text = pytesseract.image_to_string(converted_imageChi, lang='chi_sim')
    elif script_name == "Tamil":
        text = pytesseract.image_to_string(converted_imageTam, lang='tam')
    elif script_name == "Arabic":
        text1 = pytesseract.image_to_string(converted_imageChi, lang='chi_sim')
        text2 = pytesseract.image_to_string(converted_imageTam, lang='tam')
        text3 = pytesseract.image_to_string(converted_image, lang='eng')
        if len(text1) < len(text2)/1.5:
            print("Tamil")
            if len(text3) < 3*len(text2):
                text = text2
            else:
                print('eng')
                text = text3
        else:
            print("CHI")
            if len(text3) < len(text1):
                text = text1
            else:
                print('eng')
                text = text3
    else:
        text = pytesseract.image_to_string(converted_imageEng, lang='eng')
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)
