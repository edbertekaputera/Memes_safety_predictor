# Imported Libraries
from transformers import CLIPProcessor, CLIPModel
from .abstract import HarmfulnessPredictor
import numpy as np

# CLIP Image Predictor
class ClipImagePredictor(HarmfulnessPredictor):
	def __init__(self) -> None:
		self.list_of_words = [
			"harmful meme with racial stereotypes", 
			"harmful meme that disrespect religion", 
			"harmful meme with sexual orientation",
			"harmful meme with extreme nationalistic views",
			"harmful meme on socio-economic disparity",
			"harmful meme with ageism",
			"harmful meme with sexism",
			"harmful meme with ableism"
		]
		self.model = CLIPModel.from_pretrained("./clip-vit-base-patch32")
		self.processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
	
	def predict_proba(self, image:np.ndarray) -> float:
		harmful_proba = 0
		for word in self.list_of_words:
			inputs = self.processor(text=[word, "safe, harmless, normal meme"], images=image, return_tensors="pt", padding=True)
			outputs = self.model(**inputs)
			logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
			probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
			if probs[0, 0] > harmful_proba:
				harmful_proba = probs[0,0]
		return harmful_proba