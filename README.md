# RDGODE

## Overview

RDGODE is a Python-based framework that integrates Reaction-Diffusion Graph Convolutional Networks (RDGCN) with Neural Ordinary Differential Equations (Neural ODEs) for modeling complex spatiotemporal dynamics. This approach is particularly suited for applications involving reaction-diffusion systems and graph-structured data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- [Anaconda](https://www.anaconda.com/) (recommended for environment management)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yus516/RDGODE.git
   cd RDGODE
   ```

2. **Create and activate a virtual environment:**

   ```bash
   conda create -n rdgode_env python=3.8
   conda activate rdgode_env
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If `requirements.txt` is not provided, manually install necessary packages such as `torch`, `numpy`, `scipy`, `matplotlib`, and any others used in the project.*

## Usage

### Data Preparation

Prepare your dataset according to the format expected by the scripts. This typically involves:

- Graph structure files
- Feature matrices
- Labels or target values

*Ensure that your data is placed in the appropriate directories as referenced by the scripts.*

### Training and Evaluation

To train the model and evaluate its performance:

```bash
python train_with_weekday_seq.py
```

*This script trains the RDGODE model using weekday sequences.*

### Running Experiments

To run specific experiments:

```bash
python exp_RDGODE_MAML.py
```

*This script conducts experiments using Model-Agnostic Meta-Learning (MAML) with the RDGODE framework.*

## Scripts

- `train_with_weekday_seq.py`: Trains the model using weekday sequences.
- `exp_RDGODE_MAML.py`: Runs experiments with MAML.
- `FDM_network_using_RDGCN.py`: Implements the RDGCN for finite difference method networks.
- `analysis.py`: Analyzes the results and performance metrics.
- `engine.py`: Contains the training engine.
- `model.py`: Defines the model architecture.
- `util.py`: Utility functions for data processing and evaluation.

*Refer to each script's docstring or comments for detailed usage instructions.*

## Citation

If you use RDGODE in your research, please cite the following paper:

*Citation details to be added once the corresponding paper is published.*

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*For any questions or issues, please open an issue on the [GitHub repository](https://github.com/yus516/RDGODE/issues).*
