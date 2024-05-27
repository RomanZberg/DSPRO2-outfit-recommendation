from torch import nn
import torch

class OutfitClassifier(nn.Module):

    @property
    def embed_dim(self):
        return self._embed_dim

    def __init__(
            self,
            dino_architecture,
            dropout_probability,
            number_of_layers,
            hidden_neuron_count,
            device
    ):
        super(OutfitClassifier, self).__init__()
        self._device = device

        backbone_archs = {
            'small': 'vits14',
            'base': 'vitb14',
            'large': 'vitl14',
            'giant': 'vitg14',
        }

        backbone_arch = backbone_archs[dino_architecture]
        backbone_name = f'dinov2_{backbone_arch}'

        self._feature_extraction_model = torch.hub.load('facebookresearch/dinov2', backbone_name).eval().to(device)

        for param in self._feature_extraction_model.parameters():
            param.requires_grad = False

        self._embed_dim = self._feature_extraction_model.embed_dim * 5

        model_elements = [
            nn.Linear(self._embed_dim, hidden_neuron_count),
            nn.ReLU()
        ]

        for layer_number in range(number_of_layers):
            model_elements.extend([
                nn.Linear(hidden_neuron_count, hidden_neuron_count),
                nn.ReLU(),
                nn.Dropout(dropout_probability)
            ])

        model_elements.extend([
            nn.Linear(hidden_neuron_count, 1),
            nn.Sigmoid()
        ])

        self._trainable_model = nn.Sequential(
            *model_elements
        )

        self._trainable_model.to(device)
        self.counter = 0

    def forward(self, X):
        number_of_rows = X.shape[0]

        with torch.no_grad():
            # Reshape X to concatenate along the batch dimension
            # New shape will be [5 * batch_size, channels, height, width]
            dino_input = X.view(number_of_rows * 5, 3, 224, 224)

            # get dino batch features => torch.Size([160, 384])
            batch_features = self._feature_extraction_model(dino_input)

            # Reshape the features to =>  [batch_size, embed_dim]
            batch_features = torch.reshape(batch_features, (int(batch_features.shape[0] / 5), self.embed_dim))
            batch_features = batch_features.to(self._device)
            batch_features.requires_grad_()

        self.counter += 1

        return self._trainable_model(batch_features)
