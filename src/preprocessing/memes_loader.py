import os
import clip
from PIL import Image
import re
from .text_extractor import TextExtractor
from .translation import TranslatorEngine

class MemesLoader:
	def __init__(self, clip_weights:str, 
			  fasttext_weights:str, 
			  translation_weights:str, 
			  image_size=224, device="cuda"):
		
		super(MemesLoader, self).__init__()
		self.image_size = image_size
		self.device = device
		self.__text_extractor = TextExtractor()
		self.__translate = TranslatorEngine(fasttext_weights_path=fasttext_weights, 
									translation_weights_path=translation_weights, 
									device=device)
		_, self.__clip_preprocess = clip.load(clip_weights, device=device, jit=False)

	def __call__(self, image_path) -> dict:
		image = Image.open(image_path).convert('RGB')

		# Extract image and script
		text, lang = self.__text_extractor(image)
		if text == "":
			text = "null"

		if lang == "latin_alpha":
			text = self.__translate(text)
		elif text != "null":
			text = self.__translate(text, lang)

		# Filtering
		filtered_text = re.sub("[^\w\d\s,.;!?'\"“]", "", text)
		if len(text) == 0 or len(filtered_text)/len(text) < 0.9:
			filtered_text = "null"
		
		# Tokenizing enhanced text
		enh_texts = clip.tokenize(f'{"a photo of $"} , {filtered_text}', context_length=77, truncate=True).to(self.device)

		# Preprocess image
		image = image.resize((self.image_size, self.image_size))
		pixel_values = self.__clip_preprocess(image).unsqueeze(dim=0).to(self.device)

		output = {
			'image': pixel_values,
			'text': enh_texts
		}	

		return output
		
