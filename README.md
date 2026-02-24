# Bioactivity Prediction from Molecular Structures

A comprehensive machine learning project for predicting molecular bioactivity using SMILES representations, combining classical machine learning and deep learning approaches with transformer-based chemical language models.

## 📋 Overview

This project implements both **classification** and **regression** models to predict molecular bioactivity from chemical structures. The dataset contains 6,341 molecular samples with computed physicochemical descriptors and canonical SMILES representations.

### Key Features
- **Dual Modeling Approach**: Classical ML (XGBoost, RandomForest, LightGBM, CatBoost) and Deep Learning (ChemBERTa-based transformers)
- **Multiple Pretrained Models**: ChemBERTa-77M-MLM, ChemBERTa-77M-MTR, ChemZinc variants
- **SMILES Augmentation**: Random SMILES generation for data augmentation
- **Molecular Fingerprints**: Extended-Connectivity Fingerprints (ECFP) integration
- **Hyperparameter Optimization**: Optuna-based tuning with visualization
- **Model Interpretability**: LIME explanations for model predictions
- **Comprehensive Statistical Analysis**: Feature importance, normality tests, correlation analysis

## 🗂️ Project Structure

```
BioActivity/
├── classification/          # Classification tasks
│   ├── DL/                 # Deep learning models (ChemBERTa variants)
│   └── ML/                 # Classical ML models
├── regression/             # Regression tasks
│   └── bioactivity_dl 8_2 - reg*.ipynb
├── Experiments/            # Experimental notebooks
├── saved_model/            # Trained model checkpoints
│   ├── best_chemberta_model/
│   ├── ChemBERTa-77M-MTR/
│   └── ml_models/
├── saved_tokenizer/        # Tokenizers for transformer models
├── Statistical Analysis/   # Statistical reports and analysis
├── Images/                 # Visualizations and plots
├── Observations/           # XGBoost and other model observations
└── requirements.txt        # Python dependencies
```

## 📊 Dataset

- **Samples**: 6,341 molecular compounds
- **Features**: 7 physicochemical descriptors
  - Molecular Weight (MW)
  - LogP (Octanol-water partition coefficient)
  - TPSA (Topological Polar Surface Area)
  - Number of H-bond Donors
  - Number of H-bond Acceptors
  - Number of Rotatable Bonds
  - Fraction of sp³ carbons
  - Ring Count
- **Target Variable**: Binary bioactivity classification (5,118 active, 1,223 inactive)
- **Representation**: Canonical SMILES strings

## 🧠 Models Implemented

### Deep Learning
- **ChemBERTa-77M-MLM**: Masked Language Model pretrained on chemical SMILES
- **ChemBERTa-77M-MTR**: Multi-task Regression pretrained model
- **ChemZinc Variants**: Models pretrained on ZINC database
- **Custom CNN**: Convolutional neural networks for molecular feature extraction
- **Fine-tuned Transformers**: Task-specific adaptations of pretrained models

### Classical Machine Learning
- XGBoost
- Random Forest
- LightGBM
- CatBoost
- Support Vector Machines (SVM)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for deep learning models)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd BioActivity
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch 2.8.0 (with CUDA 12.6 support)
- Transformers 4.57.0
- RDKit 2025.3.5 (molecular chemistry toolkit)
- Scikit-learn 1.7.1
- XGBoost 3.1.1
- LightGBM, CatBoost
- Optuna 4.5.0 (hyperparameter optimization)
- SHAP & LIME (model interpretability)
- DeepChem 2.5.0

## 📈 Model Performance

### Classification Results
- **Random Forest**: AUC = 0.899 ± 0.007
- **Logistic Regression**: AUC = 0.724 ± 0.017
- **ChemBERTa Models**: Detailed results in experiment notebooks

### Statistical Insights
- All features show statistically significant correlation with bioactivity (p < 0.001)
- Top predictive features: NumHDonors (Cohen's d = 0.641), NumRotatableBonds (d = 0.505), TPSA (d = 0.399)
- High multicollinearity detected among molecular descriptors (VIF > 10)

## 🔬 Usage

### Training a Deep Learning Model

```python
# Example: Fine-tuning ChemBERTa for bioactivity prediction
from transformers import AutoTokenizer, AutoModel
import torch

# Load pretrained model
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
model = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')

# See classification/DL/bioactivity_dl 8_2.ipynb for complete pipeline
```

### Running Classical ML Models

```python
# Example: Training XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load preprocessed features
# X = ECFP fingerprints + molecular descriptors
# y = bioactivity labels

model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

# See classification/ML/bioactivity_ml.ipynb for complete pipeline
```

### Hyperparameter Optimization

```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        # ... other hyperparameters
    }
    # Training and validation logic
    return validation_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 📊 Visualization & Analysis

The project includes:
- **Optuna Study Visualizations**: 
  - `optuna_history.html` - Optimization history
  - `optuna_parallel_coordinate.html` - Parameter relationships
  - `optuna_param_importances.html` - Feature importance
- **LIME Explanations**: `lime_explanation.html` - Model interpretability
- **Statistical Report**: `statistical_analysis_report.txt` - Comprehensive dataset analysis

## 🔑 Key Notebooks

### Classification
- `bioactivity_dl 8_2.ipynb` - Main ChemBERTa classification pipeline
- `bioactivity_dl 8_2 - ChemMTR.ipynb` - Multi-task regression model
- `bioactivity_dl 8_2 - ChemZinc.ipynb` - ZINC-pretrained model
- `bioactivity_ml.ipynb` - Classical ML comparison

### Regression
- `bioactivity_dl 8_2 - reg.ipynb` - Continuous bioactivity prediction
- `bioactivity_dl 8_2 - reg - ChemMLM.ipynb` - MLM-based regression

### Ablation Studies
- `bioactivity_dl 8_2.1_woA.ipynb` - Without augmentation
- `bioactivity_dl 8_2.1_woD.ipynb` - Without descriptors
- `bioactivity_dl 8_2.1_woF.ipynb` - Without fingerprints

## 🏆 Results & Artifacts

Trained models and checkpoints are saved in:
- `saved_model/` - PyTorch model states (`.pth`, `.pt` files)
- `saved_model/best_chemberta_model/` - Best ChemBERTa checkpoint
- `saved_model/ml_models/` - Classical ML models (pickled)

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{bioactivity_prediction,
  title={Molecular Bioactivity Prediction using ChemBERTa and Classical ML},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/BioActivity}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [DeepChem](https://github.com/deepchem/deepchem) for ChemBERTa pretrained models
- [RDKit](https://www.rdkit.org/) for molecular chemistry toolkit
- [Hugging Face Transformers](https://huggingface.co/transformers/) for transformer architectures

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com]

---

**Note**: This project requires significant computational resources for training deep learning models. GPU with CUDA support is highly recommended.
