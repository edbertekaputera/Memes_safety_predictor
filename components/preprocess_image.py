from PIL import Image
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import cv2

"""Class to process images so that text retrieval is easier"""
class PreprocessImage:
    def __init__(self, metrics=['grayscale','remove_noise','thresholding','dilate','erode','opening','bilateral']):
        self.metrics = metrics
    
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        return cv2.medianBlur(image,3)
    
    def thresholding(self, image):
        # return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
        return cv2.threshold(image, 170, 255, 1)[1]
   
    def dilate(self, image):
        kernel = np.ones((3,3),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
   
    def erode(self, image):
        kernel = np.ones((3,3),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)
    
    def opening(self, image):
        kernel = np.ones((3,3),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def bilateral(self, image):
        return cv2.bilateralFilter(image,5,55,60)
    
    

    def transform_image(self, image):
        # Convert image to numpy array
        image_np = np.array(image)

        # Apply transformations based on the list of metrics
        for metric in self.metrics:
            if metric == 'grayscale':
                image_np = self.get_grayscale(image_np)
            elif metric == 'remove_noise':
                image_np = self.remove_noise(image_np)
            elif metric == 'thresholding':
                image_np = self.thresholding(image_np)
            elif metric == 'dilate':
                image_np = self.dilate(image_np) 
            elif metric == 'erode':
                image_np = self.erode(image_np)
            elif metric == 'opening':
                image_np = self.opening(image_np)
            elif metric == 'bilateral':
                image_np = self.bilateral(image_np)

        return image_np