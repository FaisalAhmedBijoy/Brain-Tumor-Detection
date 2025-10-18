import os
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTConfig
import torchvision.transforms as transforms

class BrainTumorPredictor:
    def __init__(self, model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load the model directly using transformers
        self.model = ViTForImageClassification.from_pretrained(
            model_dir,
            num_labels=4,  # Explicitly specify 4 classes
            ignore_mismatched_sizes=True  # This is important for loading
        )
        self.model.to(device)
        self.model.eval()
        
        # Define class labels
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path, return_probs=False):
        """
        Predict tumor type for a single image
        Args:
            image_path (str): Path to the image file
            return_probs (bool): If True, return probability distribution
        Returns:
            predicted class and probabilities (if return_probs=True)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            # Get predicted class
            _, predicted = torch.max(logits, 1)
            predicted_class = self.classes[predicted.item()]
            
            if return_probs:
                # Get probability distribution
                probs_dict = {cls: prob.item() for cls, prob in zip(self.classes, probs[0])}
                return predicted_class, probs_dict
            
            return predicted_class


if __name__ == "__main__":
     # Example usage
    model_dir = "checkpoints/best_model"  # Path to saved model
    predictor = BrainTumorPredictor(model_dir)
    
    # Example: predict a single image
    image_path = "data/Testing/glioma/Te-gl_0001.jpg"  # Example image path
    if os.path.exists(image_path):
        predicted_class, probabilities = predictor.predict_image(image_path, return_probs=True)
        
        print(f"\nPredicted class: {predicted_class}")
        print("\nClass probabilities:")
        for cls, prob in probabilities.items():
            print(f"{cls}: {prob:.4f}")
    else:
        print(f"Image not found at {image_path}")