# Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

## Overview
- **Source**: UCI Machine Learning Repository / Scikit-learn (`sklearn.datasets.load_breast_cancer`)
- **Creators**: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of Wisconsin)
- **Samples**: 569 instances
- **Features**: 30 numeric cell nucleus features computed from digitized images of fine needle aspirate (FNA) of breast masses
- **Classes**:
  - Malignant (M): 212 samples (~37.3%)
  - Benign (B): 357 samples (~62.7%)

## Feature Groups (10 base characteristics × 3 statistics)
1. **radius**: mean of distances from center to points on perimeter
2. **texture**: standard deviation of gray-scale values
3. **perimeter**: perimeter of cell nucleus
4. **area**: area of cell nucleus
5. **smoothness**: local variation in radius lengths
6. **compactness**: perimeter^2 / area - 1.0
7. **concavity**: severity of concave portions of the contour
8. **concave points**: number of concave portions of the contour
9. **symmetry**: nuclear symmetry
10. **fractal dimension**: "coastline approximation" - 1

Each characteristic is captured across 3 statistical aggregates:
- **Mean** (features 0-9): e.g., `mean radius`, `mean texture`
- **Standard Error (SE)** (features 10-19): e.g., `radius error`, `texture error`
- **Worst / Largest** (features 20-29): e.g., `worst radius`, `worst texture`

## Target Mapping & Positive Class Convention
- Scikit-learn native convention: `0 = Malignant`, `1 = Benign`.
- Clinical Machine Learning convention:
  - **Malignant** is mapped to positive class (`1`) for sensitivity / recall analysis.
  - **Benign** is mapped to negative class (`0`).
- The application displays clear, human-readable labels and highlights sensitivity (Recall for malignant cases) to minimize false negatives.
