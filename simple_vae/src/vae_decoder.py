from transformers.utils import logging
import torch
from torch import nn

logger = logging.get_logger(__name__)


class VaeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO allow stacks of linear layers with dropout here to better autoencode
        self.latent_to_token = nn.Linear(config.d_model, config.latent_size)
        # default 1e-9
        self.norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)

    def forward(self, latent) -> torch.Tensor:
        return self.norm(self.latent_to_token(latent))
