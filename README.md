# RumorGNN: Graph Neural Network for Rumor Detection

A Graph Neural Network (GNN) based approach for detecting rumors in social media using temporal and structural information.

## Project Structure

```
rumor_gnn/
├── data/
│   └── preprocess.py      # Data preprocessing and graph construction
├── models/
│   └── base_gnn.py        # GNN model implementations
├── train.py               # Training script
├── analysis.py            # Analysis and visualization tools
└── visualize.py           # Visualization utilities
```

## Features

- Temporal-aware graph construction
- Graph Attention Network (GAT) implementation
- Multi-head attention mechanism
- Edge feature support
- Skip connections and layer normalization
- Comprehensive visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rumor_gnn.git
cd rumor_gnn
```

2. Create a virtual environment:
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
```bash
python -m rumor_gnn.data.preprocess --data_path path/to/your/data.csv
```

2. Train the model:
```bash
python -m rumor_gnn.train
```

3. Generate visualizations:
```bash
python -m rumor_gnn.visualize
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rumorgnn2024,
  title={RumorGNN: A Graph Neural Network Approach for Rumor Detection},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Contact

For questions or suggestions, please open an issue or contact the maintainers. 