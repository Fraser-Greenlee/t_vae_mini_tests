"""
    Base transformer-VAE model.
"""
import torch
from torch import nn
from typing import Dict, Any
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.funnel.modeling_funnel import upsample
from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM

from transformer_vae.src.encoders import VAE_ENCODER_MODELS
from transformer_vae.src.decoders import VAE_DECODER_MODELS
from transformer_vae.src.vae import VAE
from transformer_vae.src.outputs import BaseTransformerVaeOutput
from transformer_vae.src.config import FunnelVaeConfig


logger = logging.get_logger(__name__)


class TransformerVae(PreTrainedModel):
    config_class = FunnelVaeConfig
    base_model_prefix = "transformer"
    global_step = None
    _calls_since_last_log = 0
    latest_logs = {
        "decoder_ce": 0,
        "seq_accuracy": 0,
        "token_accuracy": 0,
        "reg_loss_w": 0,
        "reg_loss": 0,
    }
    _last_logs: Dict[str, float] = {}

    def __init__(self, config: FunnelVaeConfig):
        super().__init__(config=config)
        funnel_transformer = AutoModelForMaskedLM.from_config(config.funnel)
        t5_transformer = AutoModelForSeq2SeqLM.from_config(config.t5)

        self.encoder = funnel_transformer.funnel.encoder
        self.decoder = t5_transformer.decoder
        self.lm_head = t5_transformer.lm_head
        self.shared_embedding = t5_transformer.shared
        self.decoder_start_token_id = self.config.t5.decoder_start_token_id

        assert (
            self.decoder_start_token_id is not None
        ), "`self.config.t5.decoder_start_token_id` has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        self.vae = VAE(
            VAE_ENCODER_MODELS[config.vae_encoder_model](self.config),
            VAE_DECODER_MODELS[config.vae_decoder_model](self.config),
            self.config.use_reg_loss,
        )

    def get_input_embeddings(self):
        return self.shared_embedding

    def set_input_embeddings(self, new_embeddings):
        self.shared_embedding = new_embeddings

    def _init_weights(self, module):
        pass

    def _regulariser_loss_weight_schedule(self):
        if self.global_step is None or not self.config.use_reg_loss:
            return 0
        # edit using https://www.desmos.com/calculator/mqzxhecfxz
        return torch.sigmoid(
            torch.tensor(self.global_step * self.config.reg_schedule_k - self.config.reg_schedule_b)
        ).item()

    def _update_logs(self, **logs):
        self._calls_since_last_log += 1
        for k, v in logs.items():
            self.latest_logs[k] = self.latest_logs.get(k, 0) + v

    def get_latest_logs(self):
        """
        Gets latest logs and refreshes the log values.

        Logs are normalised by the number of training inferences since the last log.
        """
        assert self.config.use_extra_logs
        if self._calls_since_last_log < 1:
            return {}

        result = dict(self.latest_logs)
        for k, v in result.items():
            value_increase = v - self._last_logs.get(k, 0)
            result[k] = value_increase / self._calls_since_last_log

        self._last_logs = dict(self.latest_logs)
        self._calls_since_last_log = 0

        return result

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, latent=None, **kwargs) -> Dict[str, Any]:
        """
        Should only be generating text from latent codes.
        """
        assert (
            latent is not None
        ), "Generation with Transformer-VAE's expects to be given a latent code to generate from."
        for rm_key in ["past", "attention_mask"]:
            if rm_key in kwargs:
                del kwargs[rm_key]
        return {"decoder_input_ids": input_ids, "latent": latent, **kwargs}

    def _get_encoder_outputs(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        return_dict=True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both `input_ids` and `inputs_embeds` at the same time.")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either `input_ids` or `inputs_embeds`")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.shared_embedding(input_ids)

        if self.config.gradient_checkpoint_encoder:

            def create_custom_forward(encoder):
                def custom_forward(*inputs):
                    return encoder(*inputs, False, False, False)
                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                inputs_embeds,
                attention_mask,
                token_type_ids,
            )

        import pdb
        pdb.set_trace()

        return self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=True,
        )

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.t5.decoder_start_token_id
        pad_token_id = self.config.t5.pad_token_id

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        latent=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=True,
        # unused args
        class_label=None,
        label=None,
        **unused_kwargs
    ):
        import pdb
        pdb.set_trace()
        assert return_dict, "Need return_dict=True, using tuple's is not implimented"
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None or inputs_embeds is not None:
            if decoder_input_ids is not None and input_ids.equal(decoder_input_ids) is False:
                raise ValueError(
                    "`input_ids` and `decoder_input_ids` do not match. Funnel-T5-VAE can only reproduce its input sequence."
                )
            if attention_mask is None and input_ids is not None:
                attention_mask = input_ids.ne(self.config.t5.pad_token_id).long()
            if encoder_outputs is None:
                encoder_outputs = self._get_encoder_outputs(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        if encoder_outputs is not None and (isinstance(encoder_outputs, list) or isinstance(encoder_outputs, tuple)):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        vae_outputs = self.vae(
            input_encoding=encoder_outputs.last_hidden_state if encoder_outputs and isinstance(encoder_outputs, BaseModelOutput) else None,
            latent=latent,
            attention_mask=encoder_outputs.attention_mask
        )

        if self.config.skip_upsample:
            upsampled_encoding = vae_outputs.reconstructed_encoding
        else:
            upsampled_encoding = upsample(
                vae_outputs.reconstructed_encoding,
                stride=2 ** (len(self.config.funnel.block_sizes) - 1),
                target_len=self.config.t5.n_positions,
                separate_cls=self.config.funnel.separate_cls,
                truncate_seq=self.config.funnel.truncate_seq,
            )

        # Now using T5 decoder

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels) if labels is not None else None

        decoder_ce = torch.tensor(0.0, device=upsampled_encoding.device)
        seq_accuracy = torch.tensor(0.0, device=upsampled_encoding.device)
        token_accuracy = torch.tensor(0.0, device=upsampled_encoding.device)
        decoder_outputs = None
        lm_logits = None
        if decoder_input_ids is not None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids, encoder_hidden_states=upsampled_encoding,
                use_cache=use_cache, output_hidden_states=output_hidden_states, return_dict=True,
                grad_chk_pnt_rate=self.config.decoder_grad_chk_pnt_rate
            )

            sequence_output = decoder_outputs.last_hidden_state
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.config.t5.d_model ** -0.5)
            lm_logits = self.lm_head(sequence_output)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                decoder_ce = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                chosen_tokens = torch.argmax(lm_logits, 2)
                pad_tokens = (labels == -100).int()
                correct_tokens = (chosen_tokens == labels).int() + pad_tokens
                seq_accuracy = (torch.min(correct_tokens, dim=1).values.sum() / labels.size(0)).detach()
                num_pad_tokens = pad_tokens.sum()
                token_accuracy = ((correct_tokens.sum() - num_pad_tokens) / (labels.numel() - num_pad_tokens)).detach()

        reg_loss_w = self._regulariser_loss_weight_schedule()
        loss = decoder_ce + vae_outputs.reg_loss * reg_loss_w

        if self.training and self.config.use_extra_logs:
            self._update_logs(
                decoder_ce=decoder_ce.item(), seq_accuracy=seq_accuracy, token_accuracy=token_accuracy, reg_loss=vae_outputs.reg_loss.item(), reg_loss_w=reg_loss_w
            )

        return BaseTransformerVaeOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values if decoder_outputs else None,
            decoder_hidden_states=decoder_outputs.hidden_states if decoder_outputs else None,
            hidden_states=decoder_outputs.hidden_states if decoder_outputs else None,
            decoder_attentions=decoder_outputs.attentions if decoder_outputs else None,
            cross_attentions=decoder_outputs.cross_attentions if decoder_outputs else None,
            reconstructed_encoding=vae_outputs.reconstructed_encoding,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
            latent=vae_outputs.latent,
            reg_loss=vae_outputs.reg_loss,
            decoder_ce=decoder_ce,
            seq_accuracy=seq_accuracy,
            token_accuracy=token_accuracy
        )
