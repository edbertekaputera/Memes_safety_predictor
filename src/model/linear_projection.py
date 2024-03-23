from  torch import nn

# Linear Projection Module
class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super(LinearProjection, self).__init__()
        # Define trainable projection layers
        map_layers = [nn.Linear(input_dim, output_dim),
                      nn.Dropout(p=drop_probs[0])]

        for _ in range(1, num_layers):
            map_layers.extend(
                [nn.ReLU(), nn.Linear(output_dim, output_dim), nn.Dropout(p=drop_probs[0])])

        self.proj = nn.Sequential(*map_layers)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        return self.proj(x)