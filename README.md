# RumorGNN: Graph Neural Network for Rumor Detection

A Graph Neural Network (GNN) based approach for detecting rumors in social media data. This project implements a GNN model that leverages both text content and social network structure to identify rumors.

## Features

- Graph Neural Network architecture for rumor detection
- Integration of text features and graph structure
- Temporal graph construction
- Comprehensive evaluation metrics
- Visualization tools for model analysis
- TensorBoard integration for training monitoring

## Project Structure

```
rumor_gnn/
├── data/
│   ├── preprocess.py      # Data preprocessing and graph construction
│   └── rumor_dataset.csv  # Dataset (not included in repo)
├── models/
│   ├── base_gnn.py        # Base GNN model implementations
│   └── gnn_model.py       # Main RumorGNN model
├── utils/
│   └── metrics.py         # Evaluation metrics
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── visualize.py           # Visualization tools
└── config.yaml            # Configuration file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rumor-gnn.git
cd rumor-gnn
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - Place your dataset in `data/rumor_dataset.csv`
   - The dataset should contain text content and social network information

2. Train the model:
```bash
python rumor_gnn/train.py
```

3. Evaluate the model:
```bash
python rumor_gnn/evaluate.py
```

4. Visualize results:
```bash
python rumor_gnn/visualize.py
```

## Model Architecture

The RumorGNN model combines:
- Text feature extraction using TF-IDF
- Graph Neural Network for learning graph structure
- Attention mechanisms for focusing on important nodes
- Temporal features for capturing rumor evolution

## Results

The model achieves:
- High accuracy in rumor detection
- Robust performance across different types of rumors
- Good generalization to unseen data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rumorgnn2024,
  author = {Your Name},
  title = {RumorGNN: Graph Neural Network for Rumor Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/rumor-gnn}
}
```

## Contact

For questions or suggestions, please open an issue or contact the maintainers. 