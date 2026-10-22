# Breast Cancer Detection System Using Machine Learning and Deep Learning

An academic and clinical machine learning research system developed for an MCA Artificial Intelligence and Machine Learning project. The application utilizes the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to classify fine-needle aspirate (FNA) cell nuclei as either **Benign (non-cancerous)** or **Malignant (cancerous)**.

---

## 1. Project Introduction
Early detection of breast cancer plays a decisive role in reducing mortality and improving survival rates. Conventional diagnosis combines mammography with histological examination of tissue obtained via fine-needle aspirate (FNA). This project delivers an automated computational pipeline that standardizes cytological feature vectors, feeds them into multiple supervised Machine Learning models and a Deep Artificial Neural Network (ANN), and surfaces predictions with probabilistic confidence, risk stratification, and feature contribution analysis.

---

## 2. Problem Statement
Given 30 numerical measurements extracted from digitized microscopic images of an FNA breast mass specimen, build an end-to-end classification system to predict whether the tumor is **Malignant** ($y = 1$) or **Benign** ($y = 0$). Because misclassifying a malignant tumor as benign (**False Negative**) delays critical oncological therapy, model optimization specifically prioritizes **Clinical Sensitivity (Recall for Malignant cases)** and **ROC-AUC**.

---

## 3. Project Objectives
1. **Zero Data Leakage:** Ensure preprocessing scalers and pipelines are strictly fit on training partitions only.
2. **Comprehensive ML Benchmarking:** Train, cross-validate, and evaluate 6 supervised algorithms (Logistic Regression, Support Vector Machine, Random Forest, K-Nearest Neighbors, Decision Tree, and Gradient Boosting).
3. **Deep Learning Implementation:** Construct, regularize, and evaluate an Artificial Neural Network (ANN) using TensorFlow / Keras with early stopping and checkpointing.
4. **Clinical Evaluation:** Quantify performance using Accuracy, Precision, Sensitivity/Recall, Specificity, F1-Score, and ROC-AUC.
5. **Interactive Web Application:** Provide dual access:
   - A Flask backend with server-rendered templates (`app.py`)
   - An interactive, responsive web interface on port 3000 with real-time sample loaders, multi-model evaluation, and diagnostic gauges.

---

## 4. Dataset Information
- **Source:** Wisconsin Diagnostic Breast Cancer (WDBC) available through Scikit-Learn (`sklearn.datasets.load_breast_cancer`)
- **Total Records:** 569 patient instances
- **Classes:** 
  - Malignant: 212 samples (37.3%)
  - Benign: 357 samples (62.7%)
- **Features:** 30 real-valued numerical attributes describing cell nucleus morphology:
  - 10 base characteristics: *Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension*
  - Computed across 3 statistics: **Mean (0-9)**, **Standard Error (10-19)**, and **Worst / Largest (20-29)**
- **Target Mapping:**
  - Scikit-learn raw convention: `0 = Malignant`, `1 = Benign`
  - Internal pipeline convention: `1 = Malignant (positive class)`, `0 = Benign (negative class)`

---

## 5. Technology Stack
- **Language:** Python 3.10+
- **Machine Learning:** Scikit-learn, SciPy, NumPy, Pandas
- **Deep Learning:** TensorFlow 2.x, Keras
- **Serialization:** Joblib, Keras H5/Native format
- **Data Visualization:** Matplotlib, Seaborn
- **Backend Server:** Flask (Python) / Express (Node.js runtime bridge)
- **Testing:** Pytest
- **Frontend:** HTML5, CSS3, JavaScript, Tailwind CSS, Bootstrap 5

---

## 6. System Architecture

```
                      Digitized FNA Sample (30 Features)
                                    │
                                    ▼
                     [Data Preprocessing Pipeline]
                 - Input Validation & Shape Verification
                 - StandardScaler Transformation (Zero Leakage)
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
   [Classical ML Models]                         [Deep Learning ANN]
   - Support Vector Machine (RBF)                - Dense(64, ReLU) + Dropout(0.3)
   - Logistic Regression (L2)                    - Dense(32, ReLU) + Dropout(0.2)
   - Random Forest (150 Trees)                   - Dense(1, Sigmoid)
   - Gradient Boosting                           - Binary Cross-Entropy Loss
   - K-Nearest Neighbors                         - Adam Optimizer
   - Decision Tree                               - Validation Early Stopping
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    ▼
                     [Inference & Clinical Scoring]
                      - Prediction: Benign / Malignant
                      - Calibrated Probability Score
                      - Risk Level Assessment (Low / Mod / High)
                      - Comparison Against WDBC Population Means
                                    │
                                    ▼
                    [Web Application & REST API]
                      - Flask Server (`app.py`)
                      - Single-Page Interactive Web Application
                      - Model Dashboard & ROC Curves
```

---

## 7. Model Evaluation Comparison (Held-out Test Cohort, N=114)

| Model Name | Type | Accuracy | Precision | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC | TP / FN | TN / FP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | ML | **98.25%** | **100.0%** | **95.24%** | **100.0%** | **0.9756** | **0.9947** | 40 / 2 | 72 / 0 |
| **Artificial Neural Network (ANN)** | DL | **98.25%** | **100.0%** | **95.24%** | **100.0%** | **0.9756** | **0.9937** | 40 / 2 | 72 / 0 |
| **Logistic Regression** | ML | 96.49% | 97.50% | 92.86% | 98.61% | 0.9512 | 0.9960 | 39 / 3 | 71 / 1 |
| **Random Forest** | ML | 96.49% | 100.0% | 90.48% | 100.0% | 0.9500 | 0.9947 | 38 / 4 | 72 / 0 |
| **Gradient Boosting** | ML | 96.49% | 100.0% | 90.48% | 100.0% | 0.9500 | 0.9947 | 38 / 4 | 72 / 0 |
| **K-Nearest Neighbors** | ML | 95.61% | 97.44% | 90.48% | 98.61% | 0.9383 | 0.9825 | 38 / 4 | 71 / 1 |
| **Decision Tree** | ML | 91.23% | 94.44% | 80.95% | 97.22% | 0.8718 | 0.9501 | 34 / 8 | 70 / 2 |

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10 or compatible
- Node.js 18+ (for development server)
- Git

### Linux / macOS Setup
```bash
# 1. Clone repository
git clone <repo-url>
cd breast_cancer_detection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

### Windows (CMD / PowerShell) Setup
```cmd
:: 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

:: 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Model Training & Evaluation

To train all 6 machine learning models, train the deep neural network with early stopping, and generate all evaluation plots and comparison tables:

```bash
# Step 1: Run ML models training and 5-fold cross-validation
python -m src.train_ml_models

# Step 2: Train Deep Learning ANN with Keras
python -m src.train_ann

# Step 3: Run comprehensive evaluation and generate ROC curves / confusion matrices
python -m src.evaluate_models
```

Generated artifacts:
- `models/best_model.joblib`: Top-performing classical ML model (SVM)
- `models/ann_model.keras`: Deep learning model checkpoint
- `models/preprocessing_pipeline.joblib`: Fitted `StandardScaler`
- `results/model_comparison.csv`: Metric summary table
- `results/evaluation_metrics.json`: Full evaluation metadata
- `results/*.png`: High-resolution ROC curves, confusion matrices, and training histories

---

## 10. Running Tests
Run the comprehensive Pytest verification suite:
```bash
pytest -v
```
All 11 tests verify:
- Complete dataset loading (569 samples, 30 features)
- Target label mapping consistency (1=Malignant, 0=Benign)
- Stratified 80:20 split preservation
- Scaler mathematical properties (zero mean, unit variance)
- Known benign and malignant sample classification
- ANN forward pass inference
- Strict rejection of missing, non-numeric, NaN, or non-finite inputs

---

## 11. Running the Web Application

### Option A: Standalone Flask Server
```bash
# Run Flask directly on port 5000 (or specified port)
python app.py
```
Open your browser to `http://127.0.0.1:5000`.

### Option C: Deploying to Vercel (One-Click / Git Push)
This project is pre-configured for Vercel deployment with Serverless Functions:
1. **`.python-version`** locks the Python runtime to `3.11` (preventing errors with Python 3.14).
2. **`vercel.json`** routes `/api/*` to the Python serverless function `api/index.py` and serves the frontend from `dist`.
3. **`requirements.txt`** contains lightweight, pre-compiled wheels (Scikit-Learn, NumPy, Pandas, Joblib, Flask) that deploy in seconds and comply with serverless size quotas (<250MB).
4. The Deep Learning Artificial Neural Network (ANN) uses pre-extracted weights (`models/ann_weights.joblib`) for zero-TensorFlow, sub-millisecond inference in production. Full TensorFlow training is available via `requirements-dl.txt`.

Push your changes to your Git repository:
```bash
git add .
git commit -m "Configure Vercel deployment with Python 3.11 and lightweight inference"
git push origin main
```
Vercel will automatically re-deploy successfully!

---

## 12. Project Folder Structure
```
breast_cancer_detection/
├── app.py                         # Flask backend application entry point
├── requirements.txt               # Python package dependencies
├── README.md                      # Academic documentation and setup manual
├── package.json                   # Node.js dependencies & scripts
├── vite.config.ts                 # Dev server & reverse proxy configuration
│
├── data/
│   └── README.md                  # WDBC dataset specification and feature definitions
│
├── notebooks/
│   └── breast_cancer_analysis.ipynb # Complete EDA notebook with histograms and heatmaps
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Stratified splitting and StandardScaler pipeline
│   ├── train_ml_models.py         # 6 ML algorithms with 5-fold CV & model selection
│   ├── train_ann.py               # Keras ANN training with early stopping
│   ├── evaluate_models.py         # Evaluation metrics, ROC curve & confusion matrix generator
│   └── predict.py                 # Reusable inference engine & input validator
│
├── models/
│   ├── best_model.joblib          # Selected best ML model (Support Vector Machine)
│   ├── ann_model.keras            # Saved Deep Learning ANN model
│   ├── preprocessing_pipeline.joblib # Fitted feature scaler
│   └── *.joblib                   # Individual trained ML models
│
├── templates/
│   ├── base.html                  # Responsive base layout with navigation & disclaimer
│   ├── index.html                 # Landing page with methodology highlights
│   ├── predict.html               # 30-feature input form with sample preloading
│   ├── result.html                # Diagnostic outcome card and feature insights
│   ├── dashboard.html             # Performance comparison dashboard & plots
│   └── about.html                 # Technical documentation & medical disclaimer
│
├── static/
│   ├── css/style.css              # Custom styling
│   ├── js/main.js                 # Sample loaders & validation
│   └── images/                    # Generated ROC curves, confusion matrices, and curves
│
├── results/
│   ├── model_comparison.csv       # Comparison table across all 7 models
│   ├── evaluation_metrics.json    # JSON report
│   ├── roc_curves.png             # ROC comparison plot
│   ├── confusion_matrices.png     # Test set confusion matrices
│   └── ann_training_curves.png    # Training/validation loss and accuracy curves
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py      # Tests for data loading, split, and scaling
    └── test_prediction.py         # Tests for inference, schema, and error handling
```

---

## 13. Limitations & Ethical Considerations
- **Small Sample Size:** The WDBC dataset comprises 569 cases from a single hospital center. Real-world populations encompass broader biological variance.
- **Derived Features:** Features are geometric summaries computed from segmentation software rather than direct histological image tissue.
- **Not a Diagnostic Device:** Probability values generated by classifiers represent geometric distances to decision hyperplanes, not a patient's biologically calibrated lifetime cancer risk.
- **Clinical Prerequisite:** Automated classification must be used solely as secondary decision-support for pathologists and oncologists.

---

## 14. License & Academic Attribution
Developed as an educational research prototype for the Master of Computer Applications (MCA) AI and Machine Learning curriculum. Dataset courtesy of the UCI Machine Learning Repository and the University of Wisconsin Hospitals.
