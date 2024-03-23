import torch
import torch.nn as nn
from clip.model import CLIP

PHI_INPUT_DIM = 768
class TextualInversion(nn.Module):
	def __init__(self, phi_weights_path:str, clip_model: CLIP, clip_img_enc_output_dim: int,
					drop_probs, post_dim=None, num_pre_proj_layers=1, device="cuda"):
		super(TextualInversion, self).__init__()

		self.clip_model = clip_model
		self.device = device

		# Define textual inversion network phi layers
		self.phi = nn.Sequential(
			nn.Linear(PHI_INPUT_DIM, 3072),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(3072, 3072),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(3072, PHI_INPUT_DIM)
		)

		# Add linear projection after phi
		self.phi_map = nn.Sequential(
			nn.Linear(PHI_INPUT_DIM, PHI_INPUT_DIM),
			nn.Dropout(p=drop_probs[0])
		)

		# load pre-trained weights of phi
		phi_dict = torch.load(phi_weights_path, map_location=self.device)["MLPCustom"]
		with torch.no_grad():
			self.phi[0].weight.copy_(phi_dict['layers.0.weight'])
			self.phi[0].bias.copy_(phi_dict['layers.0.bias'])
			self.phi[3].weight.copy_(phi_dict['layers.3.weight'])
			self.phi[3].bias.copy_(phi_dict['layers.3.bias'])
			self.phi[6].weight.copy_(phi_dict['layers.6.weight'])
			self.phi[6].bias.copy_(phi_dict['layers.6.bias'])

		for _, p in self.phi.named_parameters():
			p.requires_grad_(False)

		# Define linear projection after image encoder
		in_dim = clip_img_enc_output_dim
		pre_inversion_layers = [nn.Linear(in_dim, PHI_INPUT_DIM),
								nn.Dropout(p=drop_probs[0])]
		for _ in range(1, num_pre_proj_layers):
			pre_inversion_layers.extend(
				[nn.ReLU(), nn.Linear(PHI_INPUT_DIM, PHI_INPUT_DIM), nn.Dropout(p=drop_probs[0])])
		self.pre_inversion_map = nn.Sequential(*pre_inversion_layers)

		# define linear projection after clip text encoder
		self.post_inversion_map = nn.Sequential(
			nn.Linear(self.clip_model.token_embedding.embedding_dim, post_dim),
			nn.Dropout(p=drop_probs[0])
		)

	def encode_with_vstar(self, clip_model: CLIP, text: torch.tensor, v_star: torch.tensor, num_vstar=1,
							pooling=True, token_id=259, proj=True):
		x = clip_model.token_embedding(text).type(clip_model.dtype)
		_, counts = torch.unique((text == token_id).nonzero(as_tuple=True)[0], return_counts=True)
		cum_sum = torch.cat((torch.zeros(1, device=self.device).int(), torch.cumsum(counts, dim=0).to(self.device)[:-1]))
		first_vstar_indexes = (text == token_id).nonzero()[cum_sum][:, 1]
		rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_vstar)])

		if v_star.shape[0] == x.shape[0]:
			if len(v_star.shape) == 2:
				v_star = v_star.unsqueeze(1)
			x[torch.arange(x.shape[0]).repeat_interleave(num_vstar).reshape(x.shape[0],
																			num_vstar), rep_idx.T] = v_star.to(
				x.dtype)
		else:
			raise ValueError()

		x = x + clip_model.positional_embedding.type(clip_model.dtype)
		x = x.permute(1, 0, 2)
		x = clip_model.transformer(x)
		x = x.permute(1, 0, 2)
		x = clip_model.ln_final(x).type(clip_model.dtype)

		if pooling:
			if proj:
				x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
			else:
				x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
		return x

	def __call__(self, *args, **kwargs):
		return super().__call__(*args, **kwargs)

	def forward(self, prompt, image_features):
		img_features = self.pre_inversion_map(image_features)
		# img_features = F.normalize(img_features, p=2, dim=1)

		v_star = self.phi(img_features)

		v_star = self.phi_map(v_star)
		# v_star = F.normalize(v_star, p=2, dim=1)

		text_input = prompt

		features = self.encode_with_vstar(self.clip_model, text_input, v_star).float()
		# features = F.normalize(features, p=2, dim=1)

		features = self.post_inversion_map(features)
		# features = F.normalize(features, p=2, dim=1)

		return features