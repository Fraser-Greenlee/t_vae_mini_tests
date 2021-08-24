import torch
from torch.nn import functional as F
from transformers.trainer import Trainer

from simple_vae.src.reg_loss import REG_LOSSES


class VaeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_loss = REG_LOSSES[self.args.reg_loss]

    def reg_weight(self):
        if self.global_step is None or self.args.dont_use_reg_loss:
            return 0

        return torch.sigmoid(
            torch.tensor(self.global_step * self.args.reg_schedule_k - self.args.reg_schedule_b)
        ).item()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if not self.args.no_reg_loss:
            reg_loss = self.reg_loss(outputs.latent)
            loss += self.reg_weight() * reg_loss

        if self.args.recon_loss:
            loss += F.mse_loss(outputs["reconstructed_encoding"], outputs["encoder_last_hidden_state"])

        return (loss, outputs) if return_outputs else loss