from transformers.utils import logging
import torch
from torch import nn

logger = logging.get_logger(__name__)


class LatentDecoderNTokens(nn.Module):
    '''
        Take several latent tokens and convert them each full token hidden states.
    '''
    def __init__(self, config):
        super().__init__()
        self.latent_size = config.latent_size
        if self.latent_size == config.t5.d_model:
            logger.warning('Latent decoder is not rescaling the latent code.')
            self.latent_to_token = lambda x: x
        else:
            self.latent_to_token = nn.Linear(self.latent_size, config.t5.d_model)

    def forward(self, latent) -> torch.Tensor:
        return self.latent_to_token(latent)


class LatentDecoderLayerNorm(LatentDecoderNTokens):
    '''
        Use Funnel norm.
    '''
    def __init__(self, config):
        super().__init__(config)
        self.norm = nn.LayerNorm(config.funnel.d_model, config.funnel.layer_norm_eps)

    def forward(self, latent) -> torch.Tensor:
        return self.norm(super().forward(latent))


VAE_DECODER_MODELS = {
    "": LatentDecoderLayerNorm,
}
