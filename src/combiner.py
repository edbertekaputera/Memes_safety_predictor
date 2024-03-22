import torch
import torch.nn as nn

# 		self.comb = Combiner(self.map_dim, self.comb_proj, self.comb_fusion)


class Combiner(nn.Module):
	def __init__(self, input_dim: int = 1024):
		super(Combiner, self).__init__()

		self.comb_image_proj = nn.Sequential(
			nn.Linear(input_dim, 2 * input_dim),
			nn.ReLU(),
			nn.Dropout(p=0.5)
		)

		self.comb_text_proj = nn.Sequential(
			nn.Linear(input_dim, 2 * input_dim),
			nn.ReLU(),
			nn.Dropout(p=0.5)
		)

		self.comb_shared_branch = nn.Sequential(
			nn.Linear(2 * input_dim, 4 * input_dim),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4 * input_dim, 1),
			nn.Sigmoid()
		)

		self.comb_concat_branch = nn.Sequential(
			nn.Linear(2 * input_dim, 4 * input_dim),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4 * input_dim, input_dim),
		)

	def __call__(self, *args, **kwargs):
		return super().__call__(*args, **kwargs)

	def forward(self, img_projection, post_features):
		proj_img_fea = self.comb_image_proj(img_projection)
		proj_txt_fea = self.comb_text_proj(post_features)

		comb_features = torch.mul(proj_img_fea, proj_txt_fea)

		side_branch = self.comb_shared_branch(comb_features)
		central_branch = self.comb_concat_branch(comb_features)

		features = central_branch + ((1 - side_branch) * img_projection + side_branch * post_features)

		return features
