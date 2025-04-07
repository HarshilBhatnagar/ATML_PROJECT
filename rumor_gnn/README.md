# Rumor Detection using Graph Neural Networks

This project implements a Graph Neural Network (GNN) based approach for rumor detection on social networks, as described in the paper "Graph Neural Network based Approach for Rumor Detection on Social Networks". The implementation uses PyTorch Geometric and is designed to work with the PHEME dataset.

## Project Structure

```
./
├── rumor_gnn/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py     # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── gnn_model.py      # GNN model implementation
│   ├── __init__.py
│   ├── train.py              # Training loop and evaluation
│   ├── config.yaml           # Configuration file
│   └── README.md             # This file
├── setup.py                  # Package installation
├── dataset.csv              # PHEME dataset
└── Graph_Neural_Network_based_Approach_for_Rumor_Detection_on_Social_Networks.pdf  # Research paper
```

## Features

- GNN-based rumor detection using both GCN and GAT architectures
- Support for BERT embeddings for text features
- Configurable model architecture and training parameters
- Integration with Weights & Biases for experiment tracking
- Early stopping and model checkpointing
- Comprehensive evaluation metrics

## Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Transformers (for BERT)
- NetworkX
- scikit-learn
- wandb (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Ensure your dataset is in place:
   - The PHEME dataset (`dataset.csv`) should be in the project root
   - The path can be configured in `rumor_gnn/config.yaml`

2. Configure the model:
   - Edit `rumor_gnn/config.yaml` to set hyperparameters
   - Choose between GCN and GAT architectures
   - Configure training parameters

3. Train the model:
```bash
cd rumor_gnn
python train.py
```

4. Monitor training:
   - Progress is logged to console
   - If wandb is enabled, view metrics in the wandb dashboard
   - Checkpoints are saved in the `checkpoints` directory

## Configuration

The `config.yaml` file contains all configurable parameters:

- Data configuration (paths, splits)
- Model architecture (layers, dimensions, dropout)
- Training parameters (learning rate, epochs, batch size)
- Optimization settings
- Logging and saving options
- Evaluation metrics

## Model Architecture

The model implements a GNN with the following features:

- Multiple GCN/GAT layers with configurable dimensions
- Batch normalization
- Dropout for regularization
- Attention mechanism (when using GAT)
- Configurable number of attention heads

## Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{rumor-gnn,
  title={Graph Neural Network based Approach for Rumor Detection on Social Networks},
  author={Your Name},
  booktitle={Conference Name},
  year={2023}
}
```

## Acknowledgments

- The PHEME dataset
- PyTorch Geometric team
- The original paper authors 