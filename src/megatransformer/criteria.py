from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import torch.nn.functional as F

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
