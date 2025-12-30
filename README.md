
# Egyptian Hieroglyphic Few-Shot Learning with HPGN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for few-shot classification of Egyptian hieroglyphic glyphs using Hierarchical Prototype Graph Networks (HPGN) and Prototypical Networks.

## Overview

This project addresses the challenge of classifying rare Egyptian hieroglyphic symbols using few-shot learning techniques. We Introduce a new architecture:

- **HPGN (Hierarchical Prototype Graph Networks)**: Advanced architecture combining multi-scale CNN encoding, patch-level prototypes, and graph neural networks

### Key Features

- Advanced data augmentation pipeline for hieroglyphic images
- Meta-learning framework with episodic training
- Comprehensive evaluation with bootstrap confidence intervals
- Prototype visualization and interpretability tools
- Robust training with early stopping and regularization

## Notebook
Available on Kaggle (Public)
[Link Text]([https://www.kaggle.com/code/ayatollahelkolally/hpgn-80-90?scriptVersionId=288534101])

## Results

### 5-Way Classification Performance


| 1-Shot | 85.18% (CI: 84.70-85.61)
| 5-Shot | 94.84% (CI: 94.61-95.08)


*Evaluated on 20,000 episodes with 95% confidence intervals*

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/Ayatollah-Ibrahim/Fewshot-for-Hieroglyph.git
cd hieroglyphic-few-shot-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=9.5.0
opencv-python>=4.7.0
matplotlib>=3.7.0
tqdm>=4.65.0
kagglehub>=0.1.0
```

## Quick Start

### Data Preparation

```bash
# Download dataset (requires Kaggle authentication)
python scripts/preprocess_data.py --data_dir /path/to/data
```

### Training

```bash
# Train HPGN
python scripts/train.py --config configs/hpgn_config.yaml


### Evaluation

```bash
python scripts/evaluate.py \
  --model_path results/HPGN_*/best_model.pth \
  --n_way 5 \
  --k_shot 1 5 \
  --episodes 20000
```

## Architecture

### HPGN Components

1. **Multi-Scale CNN Encoder**: ResNet18-based feature extractor
2. **Patch Prototype Module**: Attention-based prototype generation
3. **Graph Neural Network**: GAT-based prototype refinement
4. **Distance-based Classification**: Euclidean distance to class prototypes

### Data Processing Pipeline

```
Raw Images → Preprocessing → Augmentation → Episode Sampling → Training
    ↓            ↓              ↓                ↓               ↓
 Grayscale   Contrast    Rotation/Affine    N-way K-shot    Meta-learning
 Resize      CLAHE       Color Jitter       Episodes        Optimization
```

## Reproducibility

All experiments are fully reproducible with fixed seeds:

```python
CONFIG = {
    "SEED": 42,
    "N_WAY": 5,
    "K_SHOT_LIST": [1, 5],
    "Q_QUERY": 5,
    "EPISODES_PER_EPOCH": 1000,
    "MAX_EPOCHS": 50,
    # ... full config in config.json
}
```

## Monitoring Training

Training curves and metrics are automatically saved:

```bash
# View learning curves
python -m src.utils.visualization \
  --history results/*/history.csv
```

## Methodology

### Meta-Learning Setup

- **Training**: 80% of classes (≥15 samples per class)
- **Validation**: 20% of classes for hyperparameter tuning
- **Testing**: Rare classes (<15 samples) for generalization

### Regularization Techniques

- Label smoothing (0.1)
- Mixup augmentation (α=0.2)
- Dropout (0.1-0.3)
- Weight decay (1e-4)
- Gradient clipping (max_norm=0.5)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hieroglyphic-fsl-2024,
  author = {Ayatollah Ibrahim},
  title = {Egyptian Hieroglyphic Few-Shot Learning with HPGN},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Ayatollah-Ibrahim/Fewshot-for-Hieroglyph}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.



