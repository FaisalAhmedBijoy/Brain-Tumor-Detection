import os
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
import torch.nn.init as init

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

class BrainTumorViT(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(BrainTumorViT, self).__init__()
        
        if pretrained:
            # First load the pretrained model with original num_classes
            self.vit = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                ignore_mismatched_sizes=False  # We'll handle the classifier replacement manually
            )
            
            # Replace the classifier layer
            classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
            
            # Initialize the new classifier weights properly
            init.xavier_uniform_(classifier.weight)
            init.zeros_(classifier.bias)
            
            # Replace the classifier
            self.vit.classifier = classifier
            
            print("Initialized model with pretrained weights and new classification head")
            print(f"Original classifier shape: [1000, {self.vit.config.hidden_size}]")
            print(f"New classifier shape: [{num_classes}, {self.vit.config.hidden_size}]")
        else:
            # Initialize a new ViT model from scratch
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_labels=num_classes,
                num_hidden_layers=12,
                hidden_size=768,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.vit = ViTForImageClassification(config)
            print("Initialized model from scratch")

    def freeze_base_model(self):
        """Freeze all parameters except the classifier"""
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier
        for param in self.vit.classifier.parameters():
            param.requires_grad = True
        
        print("Base model frozen, only classifier is trainable")

    def unfreeze_base_model(self):
        """Unfreeze all parameters"""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("All parameters are now trainable")

    def forward(self, x):
        outputs = self.vit(x)
        return outputs.logits

    def save_model(self, save_dir):
        """Save both model state and huggingface configuration"""
        os.makedirs(save_dir, exist_ok=True)
        self.vit.save_pretrained(save_dir)
        
    @classmethod
    def load_model(cls, model_dir, device):
        """Load model from saved directory"""
        model = cls(pretrained=False)  # Initialize without pretrained weights
        model.vit = ViTForImageClassification.from_pretrained(model_dir)
        model = model.to(device)
        return model

def get_model(device, pretrained=True):
    """Create and initialize the ViT model"""
    model = BrainTumorViT(pretrained=pretrained)
    model = model.to(device)
    return model

if __name__ == '__main__':
    
    model = get_model(device)
    print(model)