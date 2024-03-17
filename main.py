# Imported Libraries
import cv2
import torch
import sys
from components import ClipImagePredictor, HarmfulnessPredictor

# Classification function
def classify_meme(image_path:str, harmfulness_predictors_dict:dict[str, HarmfulnessPredictor], threshold=0.5):
	image = cv2.imread(image_path)
	results = {}
	for key, predictors in harmfulness_predictors_dict.items():
		results[key] = predictors.predict_proba(image)
	
	mean_proba = torch.mean(torch.tensor(list(results.values())))
	label = 1 if mean_proba > threshold else 0
	return mean_proba, label

# Main
def main():
	# Initialize predictors in ensemble
	harmfulness_predictors = {
		"Clip_Image": ClipImagePredictor(),
	}

	# Iteration loop to get new image filepath from sys.stdin:
	for line in sys.stdin:
		# IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
		image_path = line.rstrip()
		try:
			# Process the image
			proba, label = classify_meme(image_path, 
								harmfulness_predictors, 
								threshold=0.5)

			# Ensure each result for each image_path is a new line
			sys.stdout.write(f"{proba:.4f}\t{label}\n")

		except Exception as e:
			# Output to any raised/caught error/exceptions to stderr
			sys.stderr.write(str(e))


if __name__ == "__main__":
	main()
