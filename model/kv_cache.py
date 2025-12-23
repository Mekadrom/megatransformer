import torch

from typing import Optional


class KVCache:
    def __init__(self):
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None

    def reset(self):
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None

    def update(self, key: torch.Tensor, value: torch.Tensor):
        if self.key is None:
            self.key = key
        else:
            self.key = torch.cat([self.key, key], dim=2)
        if self.value is None:
            self.value = value
        else:
            self.value = torch.cat([self.value, value], dim=2)

    def __getitem__(self, idx):
        if idx == 0:
            return self.key
        elif idx == 1:
            return self.value
        else:
            raise IndexError(f"KVCache index out of range: {idx}")
        
    def size(self):
        return {
            "keys_shape": self.key.shape if self.key is not None else None,
            "values_shape": self.value.shape if self.value is not None else None
        }
    
    def __deepspeed_tensor_attributes__(self):
        return ['key', 'value']

class PreAllocatedKVCache:
    def __init__(self, max_length, batch_size, n_heads, d_queries, d_values, dtype=torch.float32, device='cuda'):
        self.key = torch.zeros((batch_size, n_heads, max_length, d_queries), dtype=dtype, device=device)
        self.value = torch.zeros((batch_size, n_heads, max_length, d_values), dtype=dtype, device=device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values

        self.position = 0

    def reset(self):
        self.key = torch.zeros((self.batch_size, self.n_heads, self.max_length, self.d_queries), dtype=self.key.dtype)
        self.value = torch.zeros((self.batch_size, self.n_heads, self.max_length, self.d_values), dtype=self.value.dtype)

    def update(self, key: torch.Tensor, value: torch.Tensor):
        if self.position + key.shape[2] > self.max_length:
            raise ValueError(f"Cannot update KVCache: position {self.position} + key length {key.shape[2]} exceeds max length {self.max_length}")

        self.key[:, :, self.position:self.position + key.shape[2], :] = key
        self.value[:, :, self.position:self.position + value.shape[2], :] = value
        self.position += key.shape[2]

    def __getitem__(self, idx):
        """Return slice that goes until the current position, non-inclusive."""
        if idx == 0:
            return self.key[:, :, :self.position, :]
        elif idx == 1:
            return self.value[:, :, :self.position, :]
        else:
            raise IndexError(f"PreAllocatedKVCache index out of range: {idx}")
        
    def size(self):
        return {
            "keys_shape": self.key.shape,
            "values_shape": self.value.shape
        }
    
    def __deepspeed_tensor_attributes__(self):
        return ['key', 'value']
