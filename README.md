# Brain Tumor Classification with Vision Transformer

This project implements a Vision Transformer (ViT) model for brain tumor classification using PyTorch. The model classifies brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor.

## Project Structure

```
Brain-Tumor-Detection/
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── main.py
├── checkpoints/
└── results/
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- Pillow
- tqdm
- matplotlib
- scikit-learn
- seaborn

## Installation

1. Create a new conda environment:
```bash
conda create -n brain_tumor python=3.8
conda activate brain_tumor
```

2. Install the required packages:
```bash
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
pip install numpy pillow tqdm matplotlib scikit-learn seaborn
```

## Usage

1. Prepare your data:
   - Place your training images in the `data/Training/` directory
   - Place your testing images in the `data/Testing/` directory
   - Each directory should have subdirectories for each class (glioma, meningioma, notumor, pituitary)

2. Train the model:
```bash
cd src
python main.py
```

The script will:
- Load and preprocess the data
- Train the Vision Transformer model
- Save the best model and checkpoints
- Generate evaluation metrics and visualizations

## Model Architecture

The model uses the Vision Transformer (ViT) architecture with the following specifications:
- Base model: ViT-B/16 (pretrained on ImageNet)
- Patch size: 16x16
- Hidden dimension: 768
- Number of heads: 12
- Number of layers: 12
- MLP size: 3072
- Classification head: 4 classes

## Results

The training script will generate:
- Model checkpoints in the `checkpoints/` directory
- Training history plot showing loss and accuracy curves
- Confusion matrix and classification report in the `results/` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.