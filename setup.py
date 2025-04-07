from setuptools import setup, find_packages
from pathlib import Path

# Get the long description from the README file
try:
    long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = "Graph Neural Network based Approach for Rumor Detection on Social Networks"

setup(
    name="rumor_gnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",  # Let pip resolve the best version
        "torch>=2.2.0",
        "torch-geometric>=2.5.0",
        "transformers>=4.37.0",
        "networkx>=3.2.1",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "umap-learn>=0.5.5",
        "captum>=0.7.0",
        "tensorboard>=2.15.0",
        "optuna>=3.5.0",
    ],
    setup_requires=[
        "numpy",  # Required for setup
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Graph Neural Network based Approach for Rumor Detection on Social Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rumor-gnn",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 