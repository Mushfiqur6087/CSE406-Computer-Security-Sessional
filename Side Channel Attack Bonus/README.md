# Website Fingerprinting Dataset Analysis and Model Training

This project implements a website fingerprinting system using deep learning models to classify network traffic traces from three different websites.

## Dataset Overview

- **Total Samples**: 56,853 traffic traces
- **Number of Classes**: 3 websites
- **Websites**:
  1. `https://cse.buet.ac.bd/moodle/` (BUET Moodle)
  2. `https://google.com` (Google)
  3. `https://prothomalo.com` (Prothom Alo News)
- **Feature Size**: 1,000 data points per trace (normalized)
- **Data Format**: JSON with normalized traffic timing sequences

## Model Architectures

### Basic Model
- **Parameters**: 1,035,011
- **Architecture**: Simple feedforward neural network
- **Purpose**: Baseline comparison model

### Complex Model
- **Parameters**: 4,161,859
- **Architecture**: More sophisticated neural network with additional layers
- **Purpose**: Enhanced performance model

## Training Configuration

- **Cross-validation**: 5-fold stratified cross-validation
- **Optimizer**: AdamW
- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Maximum Epochs**: 50
- **Early Stopping**: Patience of 5 epochs
- **Loss Function**: Cross-entropy loss

## Cross-Validation Results

### Basic Model Performance

| Fold | Validation Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Best Validation Loss |
|------|-------------------|------------------|----------------|------------------|---------------------|
| 1    | 80.21%           | 80.18%          | 80.22%         | 80.19%          | 0.4655             |
| 2    | 80.01%           | 80.00%          | 80.02%         | 79.93%          | 0.4626             |
| 3    | 80.37%           | 80.36%          | 80.38%         | 80.37%          | 0.4641             |
| 4    | 80.84%           | 81.02%          | 80.84%         | 80.91%          | 0.4490             |
| 5    | 81.04%           | 81.12%          | 81.04%         | 81.08%          | 0.4682             |

**Average Performance:**
- **Accuracy**: 80.49% ± 0.38%
- **Precision**: 80.54% ± 0.45%
- **Recall**: 80.50% ± 0.38%
- **F1-Score**: 80.49% ± 0.43%

#### Per-Class Performance (Basic Model - Average across folds):

| Website | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| BUET Moodle | 76.96% | 77.10% | 77.02% |
| Google | 75.39% | 74.66% | 75.01% |
| Prothom Alo | 88.93% | 89.46% | 89.19% |

### Complex Model Performance

| Fold | Validation Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Best Validation Loss |
|------|-------------------|------------------|----------------|------------------|---------------------|
| 1    | 80.63%           | 80.54%          | 80.64%         | 80.49%          | 0.4468             |
| 2    | 81.02%           | 81.14%          | 81.03%         | 81.03%          | 0.4540             |
| 3    | 80.97%           | 80.79%          | 80.98%         | 80.86%          | 0.4601             |
| 4    | 81.42%           | 81.44%          | 81.42%         | 81.30%          | 0.4406             |
| 5    | 80.35%           | 80.51%          | 80.36%         | 80.42%          | 0.4559             |

**Average Performance:**
- **Accuracy**: 80.88% ± 0.36%
- **Precision**: 80.88% ± 0.36%
- **Recall**: 80.88% ± 0.36%
- **F1-Score**: 80.82% ± 0.33%

#### Per-Class Performance (Complex Model - Average across folds):

| Website | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| BUET Moodle | 76.03% | 78.14% | 77.07% |
| Google | 77.10% | 74.45% | 75.70% |
| Prothom Alo | 88.51% | 90.61% | 89.55% |

## Model Comparison

| Metric | Basic Model | Complex Model | Improvement |
|--------|-------------|---------------|-------------|
| Accuracy | 80.49% ± 0.38% | 80.88% ± 0.36% | +0.39% |
| Precision | 80.54% ± 0.45% | 80.88% ± 0.36% | +0.34% |
| Recall | 80.50% ± 0.38% | 80.88% ± 0.36% | +0.38% |
| F1-Score | 80.49% ± 0.43% | 80.82% ± 0.33% | +0.33% |
| Parameters | 1,035,011 | 4,161,859 | 4× increase |

## Key Findings

1. **Best Performing Model**: Complex Model with 80.88% accuracy
2. **Most Distinguishable Website**: Prothom Alo (89.55% F1-score for complex model)
3. **Most Challenging Website**: Google (75.70% F1-score for complex model)
4. **Model Stability**: Both models show consistent performance across folds with low standard deviation
5. **Parameter Efficiency**: The complex model shows modest improvement (+0.39% accuracy) despite 4× more parameters

## Website Classification Characteristics

- **Prothom Alo**: Highest classification accuracy (~89% F1-score), likely due to distinctive traffic patterns from news website structure
- **BUET Moodle**: Moderate classification accuracy (~77% F1-score), educational platform with mixed content types
- **Google**: Most challenging to classify (~75% F1-score), possibly due to diverse services and dynamic content

## Training Insights

- **Convergence**: Models typically converged within 20-30 epochs
- **Overfitting Prevention**: Early stopping with patience=5 effectively prevented overfitting
- **Optimizer Performance**: AdamW optimizer showed stable training across all folds
- **Cross-validation Stability**: Low standard deviations indicate robust model performance

## Files Generated

- `dataset.json`: Merged and normalized dataset (56,853 samples)
- `saved_models/basic_model_fold_*.pth`: Trained basic models for each fold
- `saved_models/complex_model_fold_*.pth`: Trained complex models for each fold
- `saved_models/cross_validation_results.json`: Detailed training metrics and results

## Usage

1. **Data Preparation**: Run `merger.py` to combine individual datasets
2. **Data Validation**: Run `validate_dataset.py` to verify dataset integrity
3. **Model Training**: Run `train.py` to perform 5-fold cross-validation training
4. **Results Analysis**: Review `cross_validation_results.json` for detailed metrics

## Conclusion

The website fingerprinting system successfully achieved over 80% classification accuracy using network traffic timing patterns. The complex model provides the best performance with 80.88% accuracy, demonstrating the effectiveness of deep learning approaches for website fingerprinting tasks. The consistent cross-validation results indicate robust model generalization across different data splits.
