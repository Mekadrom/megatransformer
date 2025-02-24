from lm_eval import models, tasks
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import torch
import torch.nn.functional as F
import transformers

class LabelSmoothedCE(nn.Module):
    def __init__(self, ignore_index=-100, eps=0.1):
        super(LabelSmoothedCE, self).__init__()

        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs, targets, lengths):
        if self.eps == 0.:
            return F.cross_entropy(inputs, targets)
        
        inputs, _, _, _ = pack_padded_sequence(
            input=inputs,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        ).to(targets.device) # (sum(lengths), vocab_size)
        targets, _, _, _ = pack_padded_sequence(
            input=targets,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        ).to(targets.device) # (sum(lengths))

        n_classes = inputs.size(1)
        target_dist = (1. - self.eps) * F.one_hot(targets, n_classes).float()
        target_dist = target_dist + self.eps / n_classes
        
        log_probs = F.log_softmax(inputs, dim=1)

        loss = -(target_dist * log_probs).sum(dim=1).mean()

        return loss

# doesn't handle packing/padding sequences
class LMLoss(nn.Module):
    def __init__(self, ignore_token_id=-100, eps=0.0):
        super(LMLoss, self).__init__()
        self.ignore_token_id = ignore_token_id
        self.label_smoothing = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        loss = F.cross_entropy(
            logits, 
            labels,
            ignore_index=self.ignore_token_id,
            reduction='mean',
            label_smoothing=self.label_smoothing
        )
        
        return loss

class DecoderOnlyMoELoss(nn.Module):
    def __init__(self, diversity_loss_coefficient=0.0):
        super(DecoderOnlyMoELoss, self).__init__()

        self.diversity_loss_coefficient = diversity_loss_coefficient

    def forward(self, gating_variances: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.diversity_loss_coefficient > 0:
            moe_gating_variances_tensor = torch.stack(gating_variances).std(dim=0).mean()

            moe_diversity_loss = moe_gating_variances_tensor * self.diversity_loss_coefficient
            return moe_diversity_loss, moe_gating_variances_tensor
        else:
            return torch.tensor(0.0), torch.tensor(0.0)
        
class TransformerMoELoss(nn.Module):
    def __init__(self, diversity_loss_coefficient=0.0):
        super(TransformerMoELoss, self).__init__()

        self.diversity_loss_coefficient = diversity_loss_coefficient

    def forward(self, encoder_gating_variances: list[torch.Tensor], decoder_gating_variances: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.diversity_loss_coefficient > 0:
            encoder_moe_gating_variances_tensor = torch.stack(encoder_gating_variances).std(dim=0).mean()
            decoder_moe_gating_variances_tensor = torch.stack(decoder_gating_variances).std(dim=0).mean()
            moe_diversity_loss = (encoder_moe_gating_variances_tensor + decoder_moe_gating_variances_tensor) / 2

            moe_diversity_loss = moe_diversity_loss * self.diversity_loss_coefficient
            return moe_diversity_loss, encoder_moe_gating_variances_tensor, decoder_moe_gating_variances_tensor
        else:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

# class CustomLMEvalModel(models.LMEvalModel):
#     def __init__(self, model, tokenizer):
#         self._model = model  # Your custom PyTorch model
#         self._tokenizer = tokenizer
        
#     def _model_call(self, inputs):
#         # Convert inputs to your model's expected format
#         # Typically involves handling tokenization and attention masks
#         with torch.no_grad():
#             outputs = self._model(inputs)
#             return outputs
    
#     def generate(self, inputs, max_length, **kwargs):
#         # Implement generation logic for your specific model
#         generated = self._model.generate(
#             inputs, 
#             max_length=max_length,
#             **kwargs
#         )
#         return generated
    
#     def tok_encode(self, string):
#         return self._tokenizer.encode(string, add_special_tokens=False)
    
#     def tok_decode(self, tokens):
#         return self._tokenizer.decode(tokens)
