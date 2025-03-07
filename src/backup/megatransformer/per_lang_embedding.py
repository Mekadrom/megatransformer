from torch import nn

import torch

class PerLangEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, per_lang_embedding_layers, embedding_activation):
        super(PerLangEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.per_lang_embedding_layers = per_lang_embedding_layers
        self.embedding_activation = embedding_activation

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embeddings = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(per_lang_embedding_layers)])
        self.activation = embedding_activation

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        N, P, D = sequences.shape

        lang_indices = sequences[:, 0]

        flat_sequences = self.embed_tokens(sequences).view(-1, D)
        output = torch.zeros(N*P, self.embeddings[0].out_features, device=sequences.device)

        for i in range(len(self.embeddings)):
            embedding_mask = lang_indices == i
            embedding_input = flat_sequences[embedding_mask.any(dim=1)]
            embedding_output = self.embeddings[i](embedding_input)

            output[embedding_mask.any(dim=1)] += embedding_output

        return output.view(N, P, -1)
