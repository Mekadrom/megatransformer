from transformers import Seq2SeqTrainer, Trainer

import torch

def compute_moe_loss(moe_diversity_loss_coefficient, encoder_vars: list[torch.Tensor], decoder_vars: list[torch.Tensor]) -> torch.Tensor:
    encoder_loss = torch.stack(encoder_vars).std(dim=0).mean()
    decoder_loss = torch.stack(decoder_vars).std(dim=0).mean()
    return (encoder_loss + decoder_loss) * moe_diversity_loss_coefficient / 2

class CompositeMoESeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, moe_diversity_loss_coefficient, moe_diversity_inclusion_epoch, label_smoothing_eps: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_diversity_loss_coefficient = moe_diversity_loss_coefficient
        self.moe_diversity_inclusion_epoch = moe_diversity_inclusion_epoch

        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing_eps)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(labels=labels, **inputs)
        logits, encoder_gating_variances, decoder_gating_variances = outputs

        loss = self.loss_fn(logits[:, 1:].reshape(-1, logits.shape[-1]), labels[:, :-1].reshape(-1))

        if self.moe_diversity_loss_coefficient > 0 and self.state.epoch >= self.moe_diversity_inclusion_epoch:
            moe_loss = compute_moe_loss(self.moe_diversity_loss_coefficient, encoder_gating_variances, decoder_gating_variances)
            loss = loss + moe_loss

        return (loss, outputs) if return_outputs else loss

class CompositeMoECausalTrainer(Trainer):
    def __init__(self, moe_diversity_loss_coefficient, moe_diversity_inclusion_epoch, label_smoothing_eps: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_diversity_loss_coefficient = moe_diversity_loss_coefficient
        self.moe_diversity_inclusion_epoch = moe_diversity_inclusion_epoch

        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing_eps)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(labels=labels, **inputs)
        logits, encoder_gating_variances, decoder_gating_variances = outputs

        loss = self.loss_fn(logits[:, 1:].reshape(-1, logits.shape[-1], labels[:, :-1].reshape(-1)))

        if self.moe_diversity_loss_coefficient > 0 and self.state.epoch >= self.moe_diversity_inclusion_epoch:
            moe_loss = compute_moe_loss(self.moe_diversity_loss_coefficient, encoder_gating_variances, decoder_gating_variances)
            loss = loss + moe_loss

        return (loss, outputs) if return_outputs else loss
