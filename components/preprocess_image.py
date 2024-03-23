from PIL import Image
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import cv2

"""Class to process images so that text retrieval is easier"""


class PreprocessImage:
    def __init__(self, metrics=["greyscale", "remove_noise", "thresholding", "dilate", "erode", "opening"]):
        self.metrics = metrics

    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def remove_noise(self, image):
        return cv2.medianBlur(image, 1)

    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def dilate(self, image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def erode(self, image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def opening(self, image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def transform_image(self, image):
        # convert image
        image_np = np.array(image)
        
        #Apply appropriate transformations
        if "greyscale" in self.metrics:
            image_np = self.get_grayscale(image_np)
        if "remove_noise" in self.metrics:
            image_np = self.remove_noise(image_np)
        if "thresholding" in self.metrics:
            image_np = self.thresholding(image_np)
        if "dilate" in self.metrics:
            image_np = self.dilate(image_np)
        if "erode" in self.metrics:
            image_np = self.erode(image_np)
        if "opening" in self.metrics:
            image_np = self.opening(image_np)

        return image_np


if __name__ == "__main__":
    img_path = "../test/images/7i6bia.png"
    image = Image.open(img_path)

    preprocessor = PreprocessImage(metrics=["greyscale", "remove_noise", "thresholding"])

    image_np = preprocessor.transform_image(image)

    converted_image = Image.fromarray(image_np)
    converted_image.show()

    data = pytesseract.image_to_data()
    # text = pytesseract.image_to_string(
    #     converted_image, lang="eng", config="--oem 1 --psm 11"
    # )

    # Print the extracted text
    # print("Extracted Text:")
    # print(text)
