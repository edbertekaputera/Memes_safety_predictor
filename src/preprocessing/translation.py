# Libraries
from typing import Any
import fasttext
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

# Remove warnings
fasttext.FastText.eprint = lambda x: None

class TranslatorEngine:
	def __init__(self, fasttext_weights_path:str, translation_weights_path:str, device="cuda") -> None:
		"""
		Initialize a TranslatorEngine instance by passing in:
		1. path to FastText weights.
		2. path to translation (huggingface) weights.
		3. accelerator device (optional, default: cuda)
		"""
		self.__fast_text_model = fasttext.load_model(fasttext_weights_path)
		self.__translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_weights_path).to(device)
		self.__translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_weights_path)
		self.device = device
	
	def identify_language(self, text:str) -> str:
		"""
		Detect which langauge is the text
		Base model:
		https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
		- en : english
		- ta : tamil
		- zh : mandarin
		- ms : malay
		- id : bahasa
		Example:
		model = TranslatorEngine("lid.176.bin", "facebook/m2m100_418M")
		model.identify_language("ni hao")
		"""
		try:
			predictions = self.__fast_text_model.predict(text)
			predicted_language = predictions[0][0].split("__label__")[1]
			return predicted_language
		except ValueError as e:
			print(e)
			return "NA"
	
	def separate_tamil(self, text:str):
		"""Separates tamil to translateable substrings/sentences 
		(because the translation model tends to struggle with long tamil paragraphs.)
		"""
		splitted = text.split(" ")
		partitions = []
		subs = []
		subs_length = 0

		for word in splitted:
			word = word.strip()
			if word == "":
				pass
			# if subs_length + len(word) > 50 or (subs_length > 0 and subs[-1][-1] == "."):
			# 	partitions.append(" ".join(subs))
			# 	subs = []
			# 	subs_length = 0
			if subs_length + len(word) > 50:
				partitions.append(" ".join(subs))
				subs = []
				subs_length = 0
			elif (subs_length > 0 ):
				if len(subs[-1]) > 0 and subs[-1][-1] == ".":
					partitions.append(" ".join(subs))
					subs = []
					subs_length = 0
			subs_length += len(word)
			subs.append(word)
		if subs_length > 0:
			partitions.append(" ".join(subs))
		return partitions
	
	def translate(self, text:str, source_lang:str) -> str:
		"""
		Translate a given text
		- en : english
		- ta : tamil
		- zh : mandarin
		- ms : malay
		- id : bahasa
		Example:
		model = TranslatorEngine("lid.176.bin", "facebook/m2m100_418M")
		model.translate("你好，你好吗？", "zh")
		"""
		# Adjust source language of tokenizer
		self.__translation_tokenizer.src_lang = source_lang

		# Translating
		encoded = self.__translation_tokenizer(text, return_tensors="pt")
		encoded = {key: value.to(self.device) for key, value in encoded.items()}

		generated_tokens = self.__translation_model.generate(**encoded, 
													   forced_bos_token_id=self.__translation_tokenizer.get_lang_id("en"))
	
		return self.__translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

	def __call__(self, text:str, source_lang:str|None=None) -> str:
		"""
		Detect the language of the text and use it to translate the text into the English language.
		Example:
		model = TranslatorEngine("lid.176.bin", "facebook/m2m100_418M")
		model("你好，你好吗？")
		"""
		# Detect language if not provided source
		if source_lang == None:
			source_lang = self.identify_language(text)
	
		if source_lang == "ta":
			output = []
			for sentence in self.separate_tamil(text):
				output.append(self.translate(sentence, source_lang))
			translated_text = " ".join(output)

		elif source_lang == "en":
			return text
		
		else:
			translated_text = self.translate(text, source_lang)

		return translated_text