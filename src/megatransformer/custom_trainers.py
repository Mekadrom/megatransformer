from transformers import Seq2SeqTrainer

import torch
import torch.nn.functional as F

class CompositeMoETrainer(Seq2SeqTrainer):
    def __init__(self, moe_diversity_loss_coefficient, moe_diversity_inclusion_epoch, label_smoothing_eps: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.moe_diversity_loss_coefficient = moe_diversity_loss_coefficient
        self.moe_diversity_inclusion_epoch = moe_diversity_inclusion_epoch
        self.label_smoothing_eps = label_smoothing_eps

    def moe_criterion(self, epoch, encoder_moe_gating_variances, decoder_moe_gating_variances):
        if self.moe_diversity_loss_coefficient > 0 and epoch >= self.moe_diversity_inclusion_epoch:
            encoder_moe_gating_variances = torch.stack(encoder_moe_gating_variances).std(dim=0).mean()
            decoder_moe_gating_variances = torch.stack(decoder_moe_gating_variances).std(dim=0).mean()
            moe_diversity_loss = (encoder_moe_gating_variances + decoder_moe_gating_variances) / 2

            moe_diversity_loss = moe_diversity_loss * self.moe_diversity_loss_coefficient
            return moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances
        else:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        inputs.pop("attention_mask") # generated attention mask sucks
        outputs = model(labels=labels, **inputs)
        logits, encoder_gating_variances, decoder_gating_variances = outputs
        loss = F.cross_entropy(logits[:, 1:].transpose(1, 2), labels[:, :-1], label_smoothing=self.label_smoothing_eps)
        loss = loss + self.moe_criterion(self.state.epoch, encoder_gating_variances, decoder_gating_variances)[0]
        return (loss, outputs) if return_outputs else loss
