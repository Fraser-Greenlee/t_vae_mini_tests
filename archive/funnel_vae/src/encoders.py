from transformers.utils import logging
import torch
from torch import nn

logger = logging.get_logger(__name__)


class LatentEncoderMeanPoolTokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.t5.d_model == config.latent_size:
            logger.warn('Skipping desnse d_model -> latent layer as they have the same size.')
            self.token_to_latent_size = None
        else:
            self.token_to_latent_size = nn.Linear(config.t5.d_model, config.latent_size)
        self.tanh = nn.Tanh()

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def token_to_latent(self, embeddings):
        if self.token_to_latent_size:
            return self.token_to_latent_size(embeddings)
        return embeddings

    def forward(self, token_embeddings, attention_mask=None) -> torch.Tensor:
        return self.tanh(self.token_to_latent(self.mean_pooling(token_embeddings, attention_mask)))


VAE_ENCODER_MODELS = {
    "": LatentEncoderMeanPoolTokens,
}
