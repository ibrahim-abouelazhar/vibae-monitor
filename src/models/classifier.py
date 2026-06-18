import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNetClassifier(nn.Module):
    """
    DenseNet121 fine-tuned for vibration spectrogram classification.
    Input: (B, 1, 32, 32) grayscale spectrogram
    Internally resizes to 128×128 before DenseNet stem.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Adapt conv0: 3-channel RGB → 1-channel grayscale
        # Initialize grayscale weights as mean of the 3 RGB channels
        old_conv = base.features.conv0
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        base.features.conv0 = new_conv

        self.features = base.features  # DenseNet feature extractor
        in_features = base.classifier.in_features  # 1024 for DenseNet121

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize small spectrograms to 128×128 for DenseNet stem
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, (1, 1))
        f = torch.flatten(f, 1)
        return self.head(f)

    # ── Progressive freezing helpers ─────────────────────────────────────────

    def freeze_all(self):
        """Phase 1 — only the classification head is trainable."""
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_last_block(self):
        """Phase 2 — unfreeze denseblock4 + norm5 + head."""
        # Freeze everything first
        for p in self.features.parameters():
            p.requires_grad = False
        # Unfreeze last dense block and final batch-norm
        for module_name in ("denseblock4", "norm5"):
            block = getattr(self.features, module_name, None)
            if block is not None:
                for p in block.parameters():
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Phase 3 — full fine-tuning with very low learning rate."""
        for p in self.parameters():
            p.requires_grad = True

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def get_classifier(num_classes: int = 10) -> DenseNetClassifier:
    return DenseNetClassifier(num_classes=num_classes)
