from PIL import Image
import pytesseract
from . import preprocess_image as ppImg
from . import translate_image as trnsImg
import numpy as np

def extractText(image:Image):
	#create preprocess instances
	preprocessorBasic = ppImg.PreprocessImage(metrics=['grayscale'])
	preprocessorEng = ppImg.PreprocessImage(metrics=['bilateral','thresholding230'])
	preprocessorChi = ppImg.PreprocessImage(metrics=['remove_noise'])
	preprocessorTan = ppImg.PreprocessImage(metrics=['thresholding170'])

	#detect language
	image_np = preprocessorBasic.transform_image(image)
	converted_image = Image.fromarray(image_np)
	script_name, _ = trnsImg.detect_language(converted_image)

	# Initialize default
	text = "null"
	lang = "null"

	#select language
	if script_name in ["Han", "Hangul", "Katakana", "Japanese"]:
		image_npChi = preprocessorChi.transform_image(converted_image)
		converted_imageChi = Image.fromarray(image_npChi)

		text = pytesseract.image_to_string(converted_imageChi, lang='chi_sim', config="--psm 6")
		lang = "chi_sim"

	elif script_name == "Tamil":
		image_npTam = preprocessorTan.transform_image(converted_image)
		converted_imageTam = Image.fromarray(image_npTam)

		text = pytesseract.image_to_string(converted_imageTam, lang='tam', config="--psm 6")
		lang = "Tam"

	elif script_name == "Arabic":
		image_npEng = preprocessorEng.transform_image(converted_image)
		image_npChi = preprocessorChi.transform_image(converted_image)
		image_npTam = preprocessorTan.transform_image(converted_image)

		converted_imageEng = Image.fromarray(image_npEng)
		converted_imageChi = Image.fromarray(image_npChi)
		converted_imageTam = Image.fromarray(image_npTam)
		
		results_eng = trnsImg.detect_language(converted_imageEng)
		results_chi = trnsImg.detect_language(converted_imageChi)
		results_tam = trnsImg.detect_language(converted_imageTam)

		if results_tam[1] > results_chi[1] and results_tam[1] > results_eng[1]:
			if results_tam[0] != "Arabic":
				text = pytesseract.image_to_string(converted_imageTam, lang='tam', config="--psm 6")
				lang = "tam"
		elif results_chi[1] > results_eng[1]:
			if results_chi[0] != "Arabic":
				text = pytesseract.image_to_string(converted_imageChi, lang='chi_sim', config="--psm 6")
				lang = "chi"
		elif results_eng[0] != "Arabic":
			text = pytesseract.image_to_string(converted_imageEng, lang='eng', config="--psm 6")
			lang = "eng"

	elif script_name == "null":
		pass

	else:
		image_npEng = preprocessorEng.transform_image(converted_image)
		converted_imageEng = Image.fromarray(image_npEng)
		
		text = pytesseract.image_to_string(converted_imageEng, lang='eng', config="--psm 6")
		lang = "eng"

	return text, lang