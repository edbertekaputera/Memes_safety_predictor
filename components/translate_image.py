from PIL import Image
import pytesseract
from transformers import MarianMTModel, MarianTokenizer

from preprocess_image import PreprocessImage
import os

def load_images(directory):
    img_paths_list = []

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        img_paths_list.append(file_path)
    return img_paths_list

def convert_text(text):
    pass

if __name__ == "__main__":
    directory = "../test/images"
    img_paths_list = load_images(directory)

    preprocessor = PreprocessImage(transformations=6)

    for path in img_paths_list:
        img = Image.open(path)
        img_np = preprocessor.transform_image(img)

        converted_image = Image.fromarray(img_np)
        # converted_image.show()
        
        text = pytesseract.image_to_string(converted_image, lang='eng', config='--oem 1 --psm 6')
        convert_text(text)