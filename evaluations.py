import torch
from torch import nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(model, test_loader, device, save_dir='results'):
    """
    Evaluate the model and generate detailed metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Generate classification report
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Save classification report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    return report, cm

