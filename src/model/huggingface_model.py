from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

class HuggingFaceModel():
	def __init__(self, repo_path, device):
		self.device = device
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

		self.repo_path = repo_path

		self.model = AutoModel.from_pretrained(self.repo_path, local_files_only=True, device_map=self.device)
		self.processor = AutoProcessor.from_pretrained(self.repo_path, local_files_only=True)

	def predict_proba(self, image) -> float:
		harmful_proba = 0
		for word in self.list_of_words:
			inputs = self.processor(text=[word, "harmless, normal meme"], images=image, return_tensors="pt", padding="max_length").to(self.device)
			
			with torch.no_grad():
				outputs = self.model(**inputs)
			
			logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
			# siglip uses sigmoid
			probs = torch.sigmoid(logits_per_image) # these are the probabilities
			if probs[0][0] > harmful_proba:
				harmful_proba = probs[0][0]
		return harmful_proba