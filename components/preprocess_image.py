from PIL import Image
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import cv2

"""Class to process images so that text retrieval is easier"""
class PreprocessImage:
    def __init__(self, transformations=6):
        self.transformations = transformations
    
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        return cv2.medianBlur(image,3)
    
    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   
    def dilate(self, image):
        kernel = np.ones((3,3),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
   
    def erode(self, image):
        kernel = np.ones((3,3),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)
    
    def opening(self, image):
        kernel = np.ones((3,3),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    

    def transform_image(self, image):
        #convert image
        image_np = np.array(image)

        # Apply transformations based on self.transformations value
        if self.transformations >= 1:
            image_np = self.get_grayscale(image_np)
        if self.transformations >= 2:
            image_np = self.remove_noise(image_np)
        if self.transformations >= 3:
            image_np = self.thresholding(image_np)
        if self.transformations >= 4:
            image_np = self.dilate(image_np)
        if self.transformations >= 5:
            image_np = self.erode(image_np)
        if self.transformations >= 6:
            image_np = self.opening(image_np)
        
        return image_np

if __name__ == "__main__":
    img_path = "../test/images/7i6bia.png"
    image = Image.open(img_path)
    
    preprocessor = PreprocessImage(transformations=2)

    image_np = preprocessor.transform_image(image)

    converted_image = Image.fromarray(image_np)
    converted_image.show()
    
    text = pytesseract.image_to_string(converted_image, lang='eng', config='--oem 1 --psm 6')
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)