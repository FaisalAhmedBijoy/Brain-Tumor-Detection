import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from configuration import Config
config = Config()

class BrainTumorViT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(BrainTumorViT, self).__init__()
        
        if pretrained:
            # Load pretrained model from Hugging Face
            self.vit = ViTForImageClassification.from_pretrained(
                config.VISION_TRSANSFORMER_MODEL,
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # Important when changing number of classes
            )
        else:
            # Initialize a new ViT model from scratch
            vit_config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_labels=num_classes,
                num_hidden_layers=12,
                hidden_size=768,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.vit = ViTForImageClassification(vit_config)

    def forward(self, x):
        outputs = self.vit(x)
        return outputs.logits  # Return logits for compatibility with existing training code

def get_model(device, pretrained=True):
    """
    Create and initialize the ViT model
    Args:
        device: torch device (cuda/cpu)
        pretrained: whether to use pretrained weights from google/vit-base-patch16-224
    Returns:
        model: initialized model on specified device
    """
    model = BrainTumorViT(pretrained=pretrained)
    model = model.to(device)
    return model