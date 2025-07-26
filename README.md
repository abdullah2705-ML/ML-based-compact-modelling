# ML-based Compact Modeling

This repository contains a machine learning-based compact modeling project for semiconductor device characterization. The project implements neural network models with custom loss functions to predict device characteristics and generate IV (current-voltage) curves.

## Project Structure

```
├── src/
│   ├── data_loader.py              # Data loading and preprocessing utilities
│   ├── run_data_loader.py          # Script to run data preprocessing
│   ├── model_with_exact_loss.py    # Neural network model with custom loss
│   ├── exact_custom_loss.py        # Custom loss function implementation
│   ├── train_with_exact_loss.py    # Training script with exact loss
│   ├── plotiv.py                   # IV curve plotting and visualization
│   ├── models/                     # Trained model files
│   ├── plots/                      # Generated plots and figures
│   ├── evaluation/                 # Model evaluation results
│   └── *.csv, *.pkl               # Data files
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abdullah2705-ML/ML-based-compact-modelling.git
cd ML-based-compact-modelling
```

### 2. Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv tf-env-311

# Activate the virtual environment
# On macOS/Linux:
source tf-env-311/bin/activate

# On Windows:
# tf-env-311\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install tensorflow>=2.10.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
```

### 4. Verify Installation

```bash
# Test if TensorFlow is working
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Usage

### Step 1: Data Preprocessing

First, run the data loader to preprocess your device data:

```bash
cd src
python run_data_loader.py
```

This script will:
- Load and preprocess the device data
- Split data into training and testing sets
- Save preprocessed data as pickle files

### Step 2: Train the Model

Train the neural network model with custom exact loss function:

```bash
python train_with_exact_loss.py --term 1234 --epochs 1000 --batch_size 32
```

**Parameters:**
- `--term`: Loss function term selection (1234 for full model)
- `--epochs`: Number of training epochs (default: 1000)
- `--batch_size`: Batch size for training (default: 32)

**Available term options:**
- `1`: Term 1 only
- `12`: Terms 1 and 2
- `123`: Terms 1, 2, and 3
- `1234`: All terms (full model)

### Step 3: Generate Plots

Generate IV curves and device characteristics plots:

```bash
python plotiv.py
```

This will create various plots including:
- IV characteristics curves
- Transconductance plots
- Device parameter comparisons
- Model performance visualizations

## Output Files

After running the scripts, you'll find:

- **Models**: Trained neural network models in `src/models/`
- **Plots**: Generated figures in `src/plots/`
- **Evaluation**: Model evaluation results in `src/evaluation/`

## Key Features

- **Custom Loss Function**: Implements physics-aware loss functions for better device modeling
- **Flexible Training**: Support for different loss term combinations
- **Comprehensive Visualization**: Multiple plotting options for device characteristics
- **Data Preprocessing**: Automated data loading and preprocessing pipeline

## Troubleshooting

### Common Issues

1. **TensorFlow Installation Issues**:
   ```bash
   # If you encounter TensorFlow installation problems
   pip install tensorflow-cpu  # For CPU-only systems
   ```

2. **Memory Issues**:
   - Reduce batch size: `--batch_size 16`
   - Reduce number of epochs: `--epochs 500`

3. **Virtual Environment Issues**:
   ```bash
   # If virtual environment doesn't activate
   deactivate  # Deactivate any existing environment
   source tf-env-311/bin/activate  # Reactivate
   ```

## Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.10.0 | Deep learning framework |
| numpy | ≥1.21.0 | Numerical computing |
| pandas | ≥1.3.0 | Data manipulation |
| scikit-learn | ≥1.0.0 | Machine learning utilities |
| matplotlib | ≥3.5.0 | Plotting and visualization |
| scipy | ≥1.7.0 | Scientific computing |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers. 