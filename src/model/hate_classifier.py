# Libraries
import torch
from torch import nn
import torchmetrics
import clip
import pytorch_lightning as pl

# Local libraries
from .combiner import Combiner
from .textual_inversion import TextualInversion
from .linear_projection import LinearProjection

# Constants
CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ = 1024
    
# Hateful Meme Classifier
class HateClassifier(pl.LightningModule):
	def __init__(self, 
				clip_weights_path:str,
				text_inver_phi_weights_path:str,
				projection_embed_weights_path_1024: str,
				projection_embed_weights_path_768: str):
		
		super().__init__()

		self.acc = torchmetrics.Accuracy(task='binary')
		self.auroc = torchmetrics.AUROC(task='binary')

		# load pre-trained CLIP model
		self.clip_model, _ = clip.load(clip_weights_path, device=self.device, jit=False)

		# remove CLIP image encoder projection (textual projection must be computed again without projection product)
		self.clip_model.visual.proj = None

		# set CLIP model to float32 type
		self.clip_model.float()

		# freeze CLIP weights
		for _, p in self.clip_model.named_parameters():
			p.requires_grad_(False)

		self.image_map = LinearProjection(CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, 1, 1, [0.2, 0.4, 0.1])
		self.text_map = LinearProjection(self.clip_model.token_embedding.embedding_dim, 1024, 1, [0.2, 0.4, 0.1])
		
		self.comb = Combiner(1024)

		self.text_inv = TextualInversion(text_inver_phi_weights_path, self.clip_model, CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, [0.2, 0.4, 0.1], 1024, 1, device=self.device)

		self.image_map = LinearProjection(CLIP_IMG_ENC_OUTPUT_DIM_BEFORE_PROJ, 1024, 1, [0.2, 0.4, 0.1])

		pre_output_input_dim = 1024

		state_dict = torch.load(projection_embed_weights_path_1024, map_location=self.device)['state_dict']
		state_dict_768 = torch.load(projection_embed_weights_path_768, map_location=self.device)['state_dict']

		with torch.no_grad():
			self.image_map.proj[0].weight.copy_(state_dict['image_proj_weight'])
			self.image_map.proj[0].bias.copy_(state_dict['image_proj_bias'])
			self.text_inv.pre_inversion_map[0].weight.copy_(state_dict_768['image_proj_weight'])
			self.text_inv.pre_inversion_map[0].bias.copy_(state_dict_768['image_proj_bias'])

		# freeze projection layers
		for name, p in self.image_map.proj.named_parameters():
			p.requires_grad_(False)
		for name, p in self.text_inv.pre_inversion_map.named_parameters():
			p.requires_grad_(False)
		
		pre_output_layers = [nn.Dropout(0.4)]
		output_input_dim = pre_output_input_dim

		pre_output_layers.extend(
			[nn.Linear(pre_output_input_dim, 1024), nn.ReLU(), nn.Dropout(0.1)])
		output_input_dim = 1024

		self.pre_output = nn.Sequential(*pre_output_layers)
		self.output = nn.Linear(output_input_dim, 1)

		self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

	def compute_CLIP_features_without_proj(self, img_input):
		# CLIP image encoder projection is disabled in the init method
		image_features = self.clip_model.visual(img_input.type(self.clip_model.dtype))
		return image_features
	
	def forward(self, batch):
		image_features = self.compute_CLIP_features_without_proj(batch['image'])
			
		prompt = batch['text']

		txt_features = self.text_inv(prompt, image_features)
		img_projection = self.image_map(image_features)

		features = self.comb(img_projection, txt_features)

		features_pre_output = self.pre_output(features)
		logits = self.output(features_pre_output).squeeze(dim=1)  # [batch_size, 1]
		return logits

	def common_step(self, batch):
		output = {}

		logits = self.forward(batch)
		preds_proxy = torch.sigmoid(logits)
		preds = (preds_proxy >= 0.5).long()

		output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
		output['accuracy'] = self.acc(preds, batch['labels'])
		output['auroc'] = self.auroc(preds_proxy, batch['labels'])

		return output

	def training_step(self, batch, batch_idx):
		output = self.common_step(batch)

		total_loss = output['loss']

		self.log('train/total_loss', total_loss)
		self.log('train/loss', output['loss'])
		self.log('train/accuracy', output['accuracy'])
		self.log('train/auroc', output['auroc'])

		return total_loss

	def validation_step(self, batch, batch_idx):
		output = self.common_step(batch)

		total_loss = output['loss']

		self.log(f'val/total_loss', total_loss)
		self.log(f'val/loss', output['loss'])
		self.log(f'val/accuracy', output['accuracy'])
		self.log(f'val/auroc', output['auroc'])

		return total_loss

	def test_step(self, batch, batch_idx):
		output = self.common_step(batch)

		total_loss = output['loss']

		self.log(f'test/total_loss', total_loss)
		self.log(f'test/loss', output['loss'])
		self.log(f'test/accuracy', output['accuracy'])
		self.log(f'test/auroc', output['auroc'])
		return output

	def training_epoch_end(self, outputs):
		self.acc.reset()
		self.auroc.reset()

	def validation_epoch_end(self, outputs):
		self.acc.reset()
		self.auroc.reset()

	def test_epoch_end(self, outputs):
		self.acc.reset()
		self.auroc.reset()

	def configure_optimizers(self):
		param_dicts = [
			{"params": [p for n, p in self.named_parameters() if p.requires_grad]}
		]
		optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

		return optimizer


	def create_model(args):
		model = HateClassifier(args=args)
		return model
