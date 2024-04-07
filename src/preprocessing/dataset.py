from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from . import MemesLoader

class MemeDataset(Dataset):
	def __init__(self, dataframe, memes_loader: MemesLoader | None, set_type="train", fast=True, device="mps"):
		self.dataframe = dataframe[(dataframe["set"] == set_type)]
		self.meme_loader = memes_loader
		self.fast = fast
		self.device = device

	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self, idx):
		if self.fast:
			img = torch.tensor(np.load(f"./test/processed_images/{idx}.npy")).to(self.device)
			text = torch.tensor(np.load(f"./test/processed_text/{idx}.npy")).to(self.device)
			output = {
				'image': img[0],
				'text': text[0],
			}	
		else:
			impath = self.dataframe.iloc[idx]["image_path"]
			output = self.meme_loader(impath)
		output["labels"] =  self.dataframe.iloc[idx]["label"]
		return output

class MemeDataModule(LightningDataModule):
	def __init__(self, dataframe:DataFrame, clip_weights, fasttext_weights, translation_weights, device, batch_size=32, fast=True):
		super().__init__()
		self.df = dataframe
		self.batch_size = batch_size
		self.fast = fast
		self.meme_loader = None
		self.device = device
		if not fast:
			self.meme_loader = MemesLoader(clip_weights=clip_weights, 
							fasttext_weights=fasttext_weights,
							translation_weights=translation_weights,
							device=device)

	def setup(self, stage: str):
		# Assign train/val datasets for use in dataloaders
		if stage == "fit":
			self.trainDS = MemeDataset(self.df, memes_loader=self.meme_loader, set_type="train", fast=self.fast, device=self.device)
			self.valDS = MemeDataset(self.df, memes_loader=self.meme_loader, set_type="validation", fast=self.fast, device=self.device)
			
		# Assign test dataset for use in dataloader(s)
		if stage == "test":
			self.testDS = MemeDataset(self.df, memes_loader=self.meme_loader, set_type="test", fast=self.fast, device=self.device)

	def train_dataloader(self):
		return DataLoader(self.trainDS, 
							batch_size=self.batch_size,
							shuffle=True, 
							num_workers=os.cpu_count(), 
							pin_memory=True, 
							persistent_workers=True)

	def val_dataloader(self):
		return DataLoader(self.valDS, 
							batch_size=self.batch_size,
							shuffle=False, 
							num_workers=os.cpu_count(), 
							pin_memory=True, 
							persistent_workers=True)

	def test_dataloader(self):
		return DataLoader(self.testDS, 
							batch_size=self.batch_size,
							shuffle=False, 
							num_workers=os.cpu_count(), 
							pin_memory=True, 
							persistent_workers=True)