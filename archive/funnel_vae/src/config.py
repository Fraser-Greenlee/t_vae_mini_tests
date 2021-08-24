import copy
import math
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig, T5Config, FunnelConfig

from funnel_vae.src.encoders import VAE_ENCODER_MODELS
from funnel_vae.src.decoders import VAE_DECODER_MODELS
from funnel_vae.src.utils import assertEqual, assertIn

logger = logging.get_logger(__name__)


class FunnelVaeConfig(PretrainedConfig):
    r"""
    Arguments:
        latent_size (:obj:`int`, `optional`, defaults to 1,000):
            Number of dimensions to use for the sequences latent code.
        funnel_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the transformer model to use as encoder & decoder.
        vae_encoder_model (:obj:`str`, `optional`, defaults to None):
            Name of the model to encode T5 hidden states into latent codes.
        vae_decoder_model (:obj:`str`, `optional`, defaults to None):
            Name of the model to decode latent codes into T5 hidden states.
        set_seq_size (:obj:`int`, `optional`, defaults to 60):
            NOTE: Every input sequence must be padded to be equal to this length.
        t5_name (:obj:`str`, `optional`, defaults to t5-base):
            Name of the Transformer model to use as a decoder.
        *** Training Args ***
        reg_schedule_k (:obj:`float`, `optional`, defaults to 0.0025):
            Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight.
        reg_schedule_b (:obj:`float`, `optional`, defaults to 6.25):
            Added to global step in sigmoid, further delays increase in regulariser loss weight.
        use_extra_logs (:obj:`bool`, `optional`, defaults to False):
            Store extra logs during each training inference.
        gradient_checkpoint (:obj:`bool`, `optional`, defaults to False):
            Checkpoint gradients in the model.
            Currently just checkpoints after the encoder + VAE
        funnel_block_sizes (:obj:`str`, `optional`, defaults to ''):
            Size of each Funnel Encoder block, sequence is halved between each block.
            Example specification: 1_1_1
    """
    model_type = "transformer_vae"
    is_composition = True

    def __init__(
        self,
        latent_size=1_000,
        funnel_name="funnel-transformer/intermediate",
        t5_name="t5-base",
        vae_encoder_model='',
        vae_decoder_model='',
        set_seq_size=60,
        decoder_start_token_id=0,
        dont_use_reg_loss=False,
        reg_schedule_k=0.0025,
        reg_schedule_b=6.25,
        use_extra_logs=False,
        cache_dir=None,
        n_latent_tokens=5,  # set to -1 for full sequence
        funnel_block_sizes='',
        num_decoder_layers=0,
        num_decoder_heads=0,
        gradient_checkpoint_encoder=False,
        decoder_grad_chk_pnt_rate=0,
        skip_upsample=False,
        **kwargs,
    ):
        assertIn(vae_encoder_model, VAE_ENCODER_MODELS.keys(), "Unexpected VAE encoder.")
        assertIn(vae_decoder_model, VAE_DECODER_MODELS.keys(), "Unexpected VAE decoder.")

        super().__init__(**kwargs)

        self.set_seq_size = set_seq_size

        # VAE
        self.vae_encoder_model = vae_encoder_model
        self.vae_decoder_model = vae_decoder_model
        if set_seq_size < n_latent_tokens:
            logger.warning(f'set_seq_size size is smaller than n_latent_tokens, now using n_latent_tokens={set_seq_size} from {n_latent_tokens}')
            n_latent_tokens = set_seq_size
        self.latent_size = latent_size
        self.n_latent_tokens = n_latent_tokens
        self.skip_upsample = skip_upsample

        # funnel encoder model
        if 'funnel' not in kwargs:
            self.funnel = AutoConfig.from_pretrained(funnel_name, cache_dir=cache_dir)
            if funnel_block_sizes:
                self.funnel.block_sizes = [int(i) for i in funnel_block_sizes.split('_')]
            self.funnel.decoder_start_token_id = decoder_start_token_id
            self.funnel.n_positions = set_seq_size
        else:
            self.funnel = FunnelConfig(**kwargs.pop('funnel'))
        pooling_division = 2 ** (len(self.funnel.block_sizes) - 1)
        self.encoded_seq_size = math.ceil(self.funnel.n_positions / pooling_division)
        self.gradient_checkpoint_encoder = gradient_checkpoint_encoder

        # T5 decoder model
        if 't5' not in kwargs:
            self.t5 = AutoConfig.from_pretrained(t5_name, cache_dir=cache_dir)
            if num_decoder_layers:
                self.t5.num_layers = num_decoder_layers
            if num_decoder_heads:
                self.t5.num_heads = num_decoder_heads
            self.t5.decoder_start_token_id = decoder_start_token_id
            self.t5.n_positions = self.funnel.n_positions
            assertEqual(self.t5.model_type, "t5", "Need t5 model type for transformer_decoder.")
        else:
            self.t5 = T5Config(**kwargs.pop('t5'))
        self.decoder_grad_chk_pnt_rate = decoder_grad_chk_pnt_rate

        # extra training losses
        self.use_reg_loss = not dont_use_reg_loss
        if dont_use_reg_loss:
            logger.warning("Regularisation loss is turned off, you are training an Autoencoder (not a VAE).")
        self.reg_schedule_k = reg_schedule_k
        self.reg_schedule_b = reg_schedule_b
        self.use_extra_logs = use_extra_logs

        # misc
        self.use_cache = getattr(self.funnel, "use_cache", False)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["funnel"] = self.funnel.to_dict()
        output["model_type"] = self.__class__.model_type
        output['t5'] = self.t5.to_dict()
        return output
