# Libraries
from .image_preprocessor import PreprocessImage
from PIL import Image
import pytesseract
import re

# Text Extractor class
class TextExtractor:
	def __init__(self) -> None:
		"""Initializes a TextExtractor instance"""

		# create preprocess instances for each character's settings
		self.__grayscale = PreprocessImage(metrics=['grayscale'])
		self.__preprocess_latin = PreprocessImage(metrics=['bilateral','thresholding230'])
		self.__preprocess_chi = PreprocessImage(metrics=['remove_noise'])
		self.__preprocess_tam = PreprocessImage(metrics=['thresholding170'])
	
	def detect_script(self, img: Image):
		"""Takes in a PIL Image, and returns the pytesseract script along with the confidence."""
		try:
			osd = pytesseract.image_to_osd(img, config="--psm 0 -c min_characters_to_try=5")
			script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
			conf = re.search("Script confidence: (\d+\.?(\d+)?)", osd).group(1)
			return script, float(conf)
		
		except Exception as e: 
			return "null", 0.0

	def extract_text(self, image: Image, script_name:str):
		"""Extracts text from a PIL Image using the specified PyTesseract Script"""

		# Initialize default
		text = "null"
		lang = "null"

		# Scripts that are usually detected for Chinese characters
		if script_name in ["Han", "Hangul", "Katakana", "Japanese"]:
			preprocessed_image_chi = self.__preprocess_chi.transform_image(image)
			preprocessed_image_chi = Image.fromarray(preprocessed_image_chi)
			text = pytesseract.image_to_string(preprocessed_image_chi, lang='chi_sim', config="--psm 6")
			lang = "zh"

		# Scripts detected for Tamil characters
		elif script_name == "Tamil":
			preprocessed_image_tam = self.__preprocess_tam.transform_image(image)
			preprocessed_image_tam = Image.fromarray(preprocessed_image_tam)
			text = pytesseract.image_to_string(preprocessed_image_tam, lang='tam', config="--psm 6")
			lang = "ta"

		# Scripts commonly detected by pytesseract which can be any of the characters
		elif script_name == "Arabic":
			# Process with Tamil settings
			preprocessed_image_tam = self.__preprocess_tam.transform_image(image)
			preprocessed_image_tam = Image.fromarray(preprocessed_image_tam)
			# Process with Chinese settings
			preprocessed_image_chi = self.__preprocess_chi.transform_image(image)
			preprocessed_image_chi = Image.fromarray(preprocessed_image_chi)	
			# Process with Latin Alphabet settings
			preprocessed_image_latin = self.__preprocess_latin.transform_image(image)
			preprocessed_image_latin = Image.fromarray(preprocessed_image_latin)	
			
			# Recalculate script after preprocessing image on each setting
			results_latin = self.detect_script(preprocessed_image_latin)
			results_chi = self.detect_script(preprocessed_image_chi)
			results_tam = self.detect_script(preprocessed_image_tam)

			# The most confident script will be the detected script
			if results_tam[1] > results_chi[1] and results_tam[1] > results_latin[1]:
				# If arabic is found again, then just assume model fails to detect properly
				if results_tam[0] != "Arabic":
					text = pytesseract.image_to_string(preprocessed_image_tam, lang='tam', config="--psm 6")
					lang = "ta"

			elif results_chi[1] > results_latin[1]:
				if results_chi[0] != "Arabic":
					text = pytesseract.image_to_string(preprocessed_image_chi, lang='chi_sim', config="--psm 6")
					lang = "zh"

			elif results_latin[0] != "Arabic":
				text = pytesseract.image_to_string(preprocessed_image_latin, lang='eng', config="--psm 6")
				lang = "latin_alpha"

		# Null just means that an error occured and the language and text returned remains null
		elif script_name == "null":
			pass
		
		# Other unhandled scripts will be assumed for latin alphabets
		else:
			preprocessed_image_latin = self.__preprocess_latin.transform_image(image)
			preprocessed_image_latin = Image.fromarray(preprocessed_image_latin)
			text = pytesseract.image_to_string(preprocessed_image_latin, lang='eng', config="--psm 6")
			lang = "latin_alpha"

		return text.replace("\u200c", "").replace("\n", " "), lang
	
	def __call__(self, image: Image) -> tuple[str]:
		"""Extract text from a PIL Image with pytesseract OCR."""
		grayscale_image = self.__grayscale.transform_image(image)
		grayscale_image = Image.fromarray(grayscale_image)

		# Detect script
		script_name, _ = self.detect_script(grayscale_image)
		text, lang = self.extract_text(grayscale_image, script_name)

		return text, lang