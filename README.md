# MegaTransformer

## Overview
MegaTransformer is a highly customizable transformer module in one package. Given a yaml config, a transformer model is constructed according to specific parameters.

Again, the benefit of using this over the builtin module in PyTorch is that this is highly configurable with the most modern implementations and can be easily modified to suit your needs.

## Features
The following papers were used as a reference for feature implementations:

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    * Origin of the transformer architecture
* [Lessons on Parameter Sharing across Layers in Transformers](https://arxiv.org/abs/2104.06022)
    * Introduces cycle-rev as the best known parameter sharing strategy
* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
    * Introduces rotary position embedding
* [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887)
    * Introduces ReZero as a weight initialization strategy
* [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
    * Shows that post-LN is better than pre-LN
* [Understanding the Difficulty of Training Transformers](https://arxiv.org/abs/2004.08249)
    * Introduces Admin as a way to stabilize training for very large models
* [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153)
    * Introduces the idea of the FFN layer in a transformer being a mixture of experts where each expert is effectively a single neuron gate
* [Grokfast: Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/abs/2405.20233)
    * Introduces the idea of a slow gradient amplifier to stabilize training
* [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)
    * Introduces the idea of using GLU or variants of GLU (e.g. Swiglu) in the FFN layer of a transformer

## Installation
To install MegaTransformer, follow these steps:
`pip install --index-url https://test.pypi.org/simple/ megatransformer`

## Usage
Here's a basic example of how to use MegaTransformer:

```python
from megatransformer import config, megatransformer

model_config = config.TransformerConfig()
model_config.load_yaml("model_config.yaml")

model = megatransformer.Transformer(model_config)
# or for autoregressive causal LM
model = megatransformer.Decoder(model_config)

# train or load model weights for evaluation/inference like any other torch module
```

See the example usage scripts under the `examples` directory for more detailed examples.
* examples/wmt14ende.py
    * Example of training a megatransformer model on the WMT14 English-German dataset using mostly huggingface builtin tools

## Contributing
please do not touch anything

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, please contact us at support@megatransformer.com.
