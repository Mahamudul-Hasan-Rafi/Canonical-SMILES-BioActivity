# Bioactivity Prediction from Molecular Structures

A comprehensive machine learning project for predicting molecular bioactivity using SMILES representations, combining classical machine learning and deep learning approaches with transformer-based chemical language models.

## 📋 Overview

This project implements both **classification** and **regression** models to predict molecular bioactivity from chemical structures. The dataset contains 6,341 molecular samples with computed physicochemical descriptors and canonical SMILES representations.

**🏆 State-of-the-Art Achievement**: Our best model achieves **95.39% AUROC** and **92.01% balanced accuracy** on bioactivity classification, significantly outperforming classical machine learning baselines.

### Key Features
- **Dual Modeling Approach**: Classical ML (XGBoost, RandomForest, LightGBM, CatBoost) and Deep Learning (Transformer-based models)
- **Multiple Pretrained Models**: IBM MoLFormer-XL-both-10pct, ChemBERTa-77M-MLM, ChemBERTa-77M-MTR, ChemZinc variants
- **Advanced Architecture**: Multi-head attention with bilinear fusion of molecular representations
- **SMILES Augmentation**: Random SMILES generation for data augmentation
- **Molecular Fingerprints**: Extended-Connectivity Fingerprints (ECFP) integration
- **Hyperparameter Optimization**: Optuna-based tuning with 50+ trials and visualization
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

### Deep Learning (State-of-the-Art)
**Best Performing Architecture** (in `bioactivity_dl 8_2 C.ipynb`):
- **Base Model**: IBM MoLFormer-XL-both-10pct - A large-scale transformer pretrained on 1.1 billion molecular structures
- **Architecture Enhancements**:
  - Multi-head attention mechanism (12 heads, 768 dims)
  - Bilinear fusion layer for multi-modal integration
  - 5 unfrozen transformer layers for fine-tuning
  - 3-layer deep classifier with dropout regularization
- **Input Modalities**: 
  - SMILES sequences (via RoBERTa tokenizer)
  - ECFP molecular fingerprints (2048-bit)
  - Physicochemical descriptors (7 features)
- **Training Strategy**:
  - Focal loss for class imbalance handling
  - OneCycleLR scheduler with warmup
  - RMSprop optimizer
  - Gradient accumulation (2 steps)
  - Early stopping based on validation AUROC

**Other Transformer Variants**:
- **IBM MoLFormer-XL-both-10pct**: State-of-the-art 12-layer transformer with rotary positional embeddings (used in best model)
- **ChemBERTa-77M-MLM**: Masked Language Model pretrained on chemical SMILES
- **ChemBERTa-77M-MTR**: Multi-task Regression pretrained model
- **ChemZinc Variants**: Models pretrained on ZINC database
- **Custom CNN**: Convolutional neural networks for molecular feature extraction

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

### State-of-the-Art Classification Results (bioactivity_dl 8_2 C)
**Best Deep Learning Model - IBM MoLFormer-XL with Multi-Head Attention & Bilinear Fusion:**
- **AUROC**: 95.39% (0.9539)
- **Balanced Accuracy**: 92.01% (0.9201)
- **Precision**: 97.12% (0.9712)
- **Recall**: 96.24% (0.9624)
- **F1-Score**: 94.60% (0.9460)
- **Matthews Correlation Coefficient (MCC)**: 0.8281

**Model Architecture:**
- 5 unfrozen transformer layers
- 12 attention heads with 768 hidden dimensions
- Bilinear fusion of SMILES embeddings, ECFP fingerprints, and molecular descriptors
- RMSprop optimizer with OneCycleLR scheduler
- 30% dropout regularization

### Regression Results (bioactivity_dl 8_2 - reg)
**Best Continuous Bioactivity Prediction:**
- **R² Score**: 0.7006 (70.06% variance explained)
- **RMSE**: 0.7055
- **MAE**: 0.4932

### Classical ML Baseline
- **Random Forest**: AUC = 0.899 ± 0.007
- **Logistic Regression**: AUC = 0.724 ± 0.017

### Statistical Insights
- All features show statistically significant correlation with bioactivity (p < 0.001)
- Top predictive features: NumHDonors (Cohen's d = 0.641), NumRotatableBonds (d = 0.505), TPSA (d = 0.399)
- High multicollinearity detected among molecular descriptors (VIF > 10)

## 🔬 Usage

### Quick Results Overview

| Model Type | Task | Best Metric | Notebook |
|------------|------|-------------|----------|
| **IBM MoLFormer-XL + Multi-head Attention** | Classification | **AUROC: 95.39%** <br> Balanced Acc: 92.01% | `bioactivity_dl 8_2 C.ipynb` |
| **Transformer Regression** | Regression | **R²: 0.70** <br> RMSE: 0.71 | `bioactivity_dl 8_2 - reg.ipynb` |
| Random Forest | Classification | AUROC: 89.90% | `bioactivity_ml.ipynb` |

### Training a Deep Learning Model

```python
# Example: Best performing IBM MoLFormer-XL model with multi-head attention
from transformers import AutoTokenizer, AutoModel
import torch

# Load pretrained IBM MoLFormer-XL model
tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
molformer = AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)

# Our best architecture uses:
# - IBM MoLFormer-XL (12-layer transformer with rotary embeddings)
# - Multi-head attention (12 heads, 768 dims)
# - Bilinear fusion of SMILES + ECFP + descriptors
# - 5 unfrozen transformer layers
# - Focal loss for class imbalance
# - RMSprop optimizer with OneCycleLR scheduler

# See classification/DL/bioactivity_dl 8_2 C.ipynb for complete implementation
# This notebook achieves 95.39% AUROC and 92.01% balanced accuracy
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
- **`bioactivity_dl 8_2 C.ipynb`** ⭐ **State-of-the-Art Model** - **95.39% AUROC, 92.01% Balanced Accuracy**
  - Advanced IBM MoLFormer-XL with multi-head attention and bilinear fusion
  - Comprehensive Optuna hyperparameter tuning (50 trials)
  - Best performing model with detailed evaluation metrics
- `bioactivity_dl 8_2.ipynb` - Main transformer-based classification pipeline
- `bioactivity_dl 8_2 - ChemMTR.ipynb` - Multi-task regression model
- `bioactivity_dl 8_2 - ChemZinc.ipynb` - ZINC-pretrained model
- `bioactivity_ml.ipynb` - Classical ML comparison

### Regression
- **`bioactivity_dl 8_2 - reg.ipynb`** ⭐ **Best Regression Results** - **R² = 0.70, RMSE = 0.71**
  - Continuous bioactivity prediction using transformer models
  - MSE loss optimization for regression tasks
- `bioactivity_dl 8_2 - reg - ChemMLM.ipynb` - MLM-based regression
- `bioactivity_dl 8_2 - reg - ChemMTR.ipynb` - Multi-task regression variant

### Ablation Studies
- `bioactivity_dl 8_2.1_woA.ipynb` - Without augmentation
- `bioactivity_dl 8_2.1_woD.ipynb` - Without descriptors
- `bioactivity_dl 8_2.1_woF.ipynb` - Without fingerprints

## 🏆 Results & Artifacts

### Best Model Checkpoints
Trained models and checkpoints are saved in:
- `saved_model/best_chemberta_model/` - **State-of-the-art classification model** (95.39% AUROC) - fine-tuned IBM MoLFormer-XL
- `saved_model/best_regression_model_full.pth` - **Best regression model** (R² = 0.70)
- `saved_model/ChemBERTa-77M-MTR/` - Alternative pretrained models
- `saved_model/ml_models/` - Classical ML models (pickled)
- `best_model_earlystop.pt`, `best_finetuned_model_chem.pth` - Additional fine-tuned checkpoints

### Performance Summary
**Classification Task:**
- 📊 AUROC: **95.39%** (validation set)
- ⚖️ Balanced Accuracy: **92.01%**
- 🎯 Precision: **97.12%** / Recall: **96.24%**
- 📈 F1-Score: **94.60%**
- 🔬 Matthews Correlation: **0.828**

**Regression Task:**
- 📐 R² Score: **0.7006** (70% variance explained)
- 📏 Root Mean Squared Error: **0.7055**
- 📌 Mean Absolute Error: **0.4932**

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

- [IBM Research](https://github.com/IBM/molformer) for the MoLFormer-XL pretrained model
- [DeepChem](https://github.com/deepchem/deepchem) for ChemBERTa pretrained models
- [RDKit](https://www.rdkit.org/) for molecular chemistry toolkit
- [Hugging Face Transformers](https://huggingface.co/transformers/) for transformer architectures

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com]

---

**Note**: This project requires significant computational resources for training deep learning models. GPU with CUDA support is highly recommended.
