import os
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class HuggingFaceLoader:
	def __init__(self, image_size=256, device="cuda"):
		
		super(HuggingFaceLoader, self).__init__()
		self.image_size = image_size
		self.device = device

	def __call__(self, image_path) -> dict:
		image = Image.open(image_path).convert('RGB')

		# Preprocess image
		image = image.resize((self.image_size, self.image_size)).to(self.device)

		return image
		
