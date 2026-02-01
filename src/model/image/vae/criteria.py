import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import lpips


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""
    def __init__(self, feature_layers=[3, 8, 15, 22], use_input_norm=True):
        """
        Args:
            feature_layers: indices of VGG layers to extract features from
                - 3: relu1_2 (64 channels)
                - 8: relu2_2 (128 channels)
                - 15: relu3_3 (256 channels)
                - 22: relu4_3 (512 channels)
            use_input_norm: whether to normalize input to ImageNet stats
        """
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # extract only the feature layers we need (slim)
        max_layer = max(feature_layers) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])
        self.feature_layers = feature_layers

        # freeze
        self.features.requires_grad_(False)
        self.features.eval()

        self.use_input_norm = use_input_norm
        if use_input_norm:
            # ImageNet normalization
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()  # always eval
        return self

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            target = (target - self.mean) / self.std

        loss = 0.0
        x_feat = x
        target_feat = target

        for i, layer in enumerate(self.features):
            x_feat = layer(x_feat)
            target_feat = layer(target_feat)

            if i in self.feature_layers:
                loss = loss + F.mse_loss(x_feat, target_feat) / len(self.feature_layers)

        return loss


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss wrapper.

    Requires: pip install lpips

    Model sizes:
        - 'alex' (AlexNet): ~9.1 MB, fastest, default
        - 'squeeze' (SqueezeNet): ~2.8 MB, smallest
        - 'vgg': ~58.9 MB, closest to traditional perceptual loss
    """
    def __init__(self, net: str = 'alex'):
        super().__init__()

        self.lpips = lpips.LPIPS(net=net, verbose=False)
        self.lpips.requires_grad_(False)
        self.lpips.eval()

    def train(self, mode=True):
        super().train(mode)
        self.lpips.eval()
        return self

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Expects inputs in [-1, 1] range."""
        loss: torch.Tensor = self.lpips(x, target)
        return loss.mean()


class DINOv2PerceptualLoss(nn.Module):
    """
    Perceptual loss using DINOv2 features.

    DINOv2 provides more semantic, less texture-focused features compared to VGG.
    This can help preserve content correctness while being less sensitive to
    exact texture patterns that can cause GAN artifacts.

    Reference: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
    """
    def __init__(
        self,
        model_name: str = "dinov2_vits14",  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14
        use_patch_tokens: bool = True,  # Use patch tokens (spatial) vs CLS token (global)
        layers: list = None,  # Which intermediate layers to use (None = final only)
    ):
        super().__init__()

        # Load DINOv2 model from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.requires_grad_(False)
        self.model.eval()

        self.use_patch_tokens = use_patch_tokens
        self.layers = layers

        # DINOv2 expects ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Patch size for DINOv2 ViT models
        self.patch_size = 14

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # Always eval
        return self

    def _normalize(self, x):
        """Convert from [-1, 1] to ImageNet normalized."""
        # First convert [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Then apply ImageNet normalization
        return (x - self.mean) / self.std

    def _resize_if_needed(self, x):
        """Resize to be divisible by patch size."""
        h, w = x.shape[-2:]
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute DINO perceptual loss.

        Args:
            x: Reconstructed images [-1, 1]
            target: Target images [-1, 1]

        Returns:
            Perceptual loss (scalar)
        """
        # Normalize and resize
        x = self._normalize(x)
        target = self._normalize(target)
        x = self._resize_if_needed(x)
        target = self._resize_if_needed(target)

        if self.use_patch_tokens:
            # Get patch token features (spatial)
            x_features = self.model.forward_features(x)
            target_features = self.model.forward_features(target)

            # Extract patch tokens (exclude CLS token)
            x_patches = x_features["x_norm_patchtokens"]
            target_patches = target_features["x_norm_patchtokens"]

            # Compute MSE loss on patch features
            loss = F.mse_loss(x_patches, target_patches)
        else:
            # Get CLS token (global feature)
            x_cls = self.model(x)
            target_cls = self.model(target)
            loss = F.mse_loss(x_cls, target_cls)

        return loss
