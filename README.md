# AlexNet on CIFAR-10

A PyTorch implementation of an AlexNet-inspired Convolutional Neural Network trained from scratch on the CIFAR-10 image classification dataset.

This project was built as part of my deep learning roadmap to understand CNN architectures by implementing and experimenting with them rather than relying on pretrained models.

## Project Highlights

- Implemented an AlexNet-inspired CNN from scratch using PyTorch
- Trained on CIFAR-10 (50,000 training images and 10,000 test images)
- Implemented data augmentation and channel-wise normalization
- Added Batch Normalization and Dropout
- Used Adam optimizer with weight decay
- Used `ReduceLROnPlateau` for adaptive learning-rate scheduling
- Conducted experiments to reduce classifier parameters while maintaining accuracy
- Achieved approximately **89.9% validation accuracy**

---

## Architecture

The model is an AlexNet-inspired architecture adapted for CIFAR-10's 32×32 RGB images.

```text
Input: 32 × 32 × 3
        │
        ▼
Conv2D: 3 → 64
BatchNorm
ReLU
MaxPool 2×2
        │
        ▼
Conv2D: 64 → 192
BatchNorm
ReLU
MaxPool 2×2
        │
        ▼
Conv2D: 192 → 384
BatchNorm
ReLU
        │
        ▼
Conv2D: 384 → 256
BatchNorm
ReLU
        │
        ▼
Conv2D: 256 → 256
BatchNorm
ReLU
MaxPool 2×2
        │
        ▼
Flatten
        │
        ▼
Fully Connected Layers
        │
        ▼
10 Class Outputs
```

### Classifier

The final version uses a reduced classifier:

```text
4096 → 1024 → 512 → 10
```

with ReLU activations and Dropout.

---

## Dataset

The model is trained on **CIFAR-10**, which contains 10 image classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Images are RGB and have a resolution of 32×32 pixels.

The dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

---

## Data Preprocessing

### Training

The training pipeline uses:

```text
Random Horizontal Flip
        ↓
Random Crop (32×32, padding=4)
        ↓
ToTensor
        ↓
Channel-wise Normalization
```

The CIFAR-10 channel statistics used for normalization are:

```text
Mean = (0.4914, 0.4822, 0.4465)
Std  = (0.2470, 0.2435, 0.2616)
```

ColorJitter was also evaluated as an additional augmentation during experimentation.

### Validation

Validation images use:

```text
ToTensor
   ↓
Normalization
```

No random augmentation is applied during validation.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Dataset | CIFAR-10 |
| Batch Size | 64 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Loss Function | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| LR Reduction Factor | 0.1 |
| Training Epochs | 30 |
| Device | CUDA if available, otherwise CPU |

---

## Experiments

Rather than only implementing the architecture, several experiments were performed to understand the effect of different design choices.

### 1. Data Augmentation

Adding ColorJitter improved validation performance compared with the initial augmentation setup.

This demonstrated the importance of improving the diversity of training examples for better generalization.

### 2. Classifier Parameter Reduction

The original classifier contained:

```text
4096 → 4096 → 10
```

The classifier was reduced to:

```text
4096 → 1024 → 512 → 10
```

The smaller classifier maintained comparable performance while significantly reducing the number of parameters in the fully connected section.

### Result

```text
Best Validation Accuracy: ~89.9%
```

This experiment demonstrated that increasing parameter count does not necessarily lead to better generalization.

---

## Training Behavior

The model initially learns with a learning rate of `0.001`.

When validation loss stops improving, `ReduceLROnPlateau` reduces the learning rate:

```text
0.001
  ↓
0.0001
  ↓
0.00001
  ↓
0.000001
```

This allows the model to make smaller parameter updates as training approaches a better solution.

---

## Project Structure

```text
AlexNet/
│
├── src/
│   ├── dataset.py      # Dataset and DataLoader
│   ├── model.py        # AlexNet architecture
│   └── train.py        # Training and validation logic
│
├── main.py             # Training entry point
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shivstar-bit/AlexNet.git
cd AlexNet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python main.py
```

The CIFAR-10 dataset will be downloaded automatically if it is not already present.

Training will automatically use CUDA when a compatible GPU is available.

---

## Model Weights

The trained model checkpoint is not stored directly in this repository to keep the Git repository focused on source code and experimentation.

Pretrained weights:

**[Download Model Weights](YOUR_GOOGLE_DRIVE_LINK)**

---

## Technologies

- Python
- PyTorch
- Torchvision
- CUDA
- CIFAR-10
- Convolutional Neural Networks

---

## Key Learning Outcomes

This project helped me understand:

- CNN feature extraction
- Convolution and pooling
- Channel dimensions
- Batch Normalization
- Dropout
- Data augmentation
- Weight decay
- Cross-entropy loss
- Backpropagation
- Adam optimization
- Learning-rate scheduling
- Overfitting and generalization
- Parameter efficiency
- CNN architecture design

---

## Future Work

This project is part of a broader CNN architecture learning roadmap.

Planned architectures include:

- VGG
- ResNet
- DenseNet
- Inception

The goal is to implement each architecture from scratch, understand the architectural decisions behind it, and experimentally compare their performance.