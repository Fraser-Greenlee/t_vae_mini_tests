from transformers.utils import logging
import torch
from torch import nn

logger = logging.get_logger(__name__)


class LatentEncoderNTokens(nn.Module):
    '''
        Converts N hidden tokens into N seperate latent codes.
    '''
    def __init__(self, config):
        super().__init__()
        self.token_to_latent = nn.Linear(config.funnel.d_model, config.latent_size)
        self.n_tokens = config.n_latent_tokens
        self.tanh = nn.Tanh()

    def forward(self, encoding, attention_mask=None) -> torch.Tensor:
        return self.tanh(self.token_to_latent(encoding))[:, : self.n_tokens, :]


class LatentEncoderMeanPoolTokens(nn.Module):
    '''
        Converts N hidden tokens into 1 latent code.
    '''
    def __init__(self, config):
        super().__init__()
        if config.t5.d_model == config.latent_size:
            logger.warn('Skipping desnse d_model -> latent layer as they have the same size.')
            self.token_to_latent_layer = None
        else:
            self.token_to_latent_layer = nn.Linear(config.t5.d_model, config.latent_size)
        self.tanh = nn.Tanh()

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def token_to_latent(self, embeddings):
        if self.token_to_latent_layer:
            return self.token_to_latent_layer(embeddings)
        return embeddings

    def forward(self, token_embeddings, attention_mask=None) -> torch.Tensor:
        return self.tanh(self.token_to_latent(self.mean_pooling(token_embeddings, attention_mask)))


VAE_ENCODER_MODELS = {
    "": LatentEncoderMeanPoolTokens,
}
