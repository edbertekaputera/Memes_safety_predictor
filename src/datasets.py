import os
import clip
from PIL import Image
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class MemesLoader(object):
	def __init__(self, image_size=224, device="cuda"):
		super(MemesLoader, self).__init__()
		self.image_size = image_size
		self.device = device
		_, self.clip_preprocess = clip.load("./resources/pretrained_weights/clip/ViT-L-14.pt", device=device, jit=False)

	def __call__(self, image_path) -> dict:
		image = Image.open(image_path).convert('RGB')

		# FILL LATER TEXT STUFF
		text = "null"
		
		# Tokenizing enhanced text
		enh_texts = clip.tokenize(f'{"a photo of $"} , {text}', context_length=77, truncate=True).to(self.device)

		# Preprocess image
		image = image.resize((self.image_size, self.image_size))
		pixel_values = self.clip_preprocess(image).unsqueeze(dim=0).to(self.device)

		output = {
			'image': pixel_values,
			'text': enh_texts
		}	

		return output
		
