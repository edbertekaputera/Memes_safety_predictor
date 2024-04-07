# Imported Libraries
import cv2
import torch
import sys

# Turn off warnings
import warnings 
warnings.warn = lambda *args,**kwargs: None

import os
# Set environment variables for cache
# os.environ["HF_HOME"] = "/app/transformer_cache"
# os.system("chmod -R 777 $HF_HOME")

from src.model import HateClassifier
from src.preprocessing import MemesLoader

# Classification function
def classify_meme(image_path, model:HateClassifier, loader:MemesLoader, threshold=0.5):
	input = loader(image_path)
	logits = model(input)[0]
	proba = torch.sigmoid(logits)
	label = (proba >= threshold).int()
	return proba, label

# Main
def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
	# Initialize 
	meme_loader = MemesLoader(clip_weights="./resources/pretrained_weights/clip/ViT-L-14.pt", 
							fasttext_weights="./resources/pretrained_weights/fasttext/lid.176.bin",
							translation_weights="./resources/pretrained_weights/facebook/m2m100_1.2B",
							device=device)

	model = HateClassifier.load_from_checkpoint("./resources/pretrained_weights/model.ckpt", 
											clip_weights_path="./resources/pretrained_weights/clip/ViT-L-14.pt",
											text_inver_phi_weights_path="./resources/pretrained_weights/phi/phi_imagenet_45.pt",
											projection_embed_weights_path_1024="./resources/pretrained_weights/hmc/hmc_1024_projection_embeddings.pt",
											projection_embed_weights_path_768="./resources/pretrained_weights/hmc/hmc_768_projection_embeddings.pt",
											map_location=device).to(device)
	model.eval()

	# Iteration loop to get new image filepath from sys.stdin:
	for line in sys.stdin:
		# IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
		image_path = line.replace("\n", "")
		image_path = image_path.rstrip()
		if image_path == "":
			continue
		try:
			# Process the image
			proba, label = classify_meme(image_path, 
								model=model, 
								loader=meme_loader,
								threshold=0.5)

			# Ensure each result for each image_path is a new line
			sys.stdout.write(f"{proba:.4f}\t{label}\n")

		except Exception as e:
			# Output to any raised/caught error/exceptions to stderr
			sys.stdout.write(f"{0.5:.4f}\t{1}\n")

if __name__ == "__main__":
	main()
