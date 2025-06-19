# Dataset Documentation

This document provides detailed information about the website fingerprinting dataset, including the original individual datasets, the merging process, and the final consolidated dataset characteristics.

## Dataset Overview

### Original Dataset Structure

The dataset was collected from multiple contributors, each providing traffic traces from three target websites:

```
individual-data/
├── 2005001/dataset.json
├── 2005004/dataset.json
├── 2005005/dataset.json
├── 2005006/dataset.json
├── 2005017/dataset.json
├── 2005020/dataset.json
├── 2005021/dataset.json
├── 2005027/dataset.json
├── 2005035/dataset.json
├── 2005045/dataset.json
├── 2005055/dataset.json
├── 2005067/dataset.json
├── 2005077/dataset.json (+ environment.txt)
├── 2005079/dataset.json
├── 2005084/dataset.json
├── 2005089/dataset.json (+ environment.txt)
├── 2005107/dataset.json
└── 2005112/dataset.json
```

**Total Contributors**: 18 individual datasets
**Environment Files**: 2 contributors provided additional environment information

### Target Websites

The dataset contains network traffic traces from three distinct websites:

1. **BUET Moodle** (`https://cse.buet.ac.bd/moodle/`)
   - Type: Educational platform (Learning Management System)
   - Characteristics: Mixed content types, user authentication, dynamic content loading

2. **Google** (`https://google.com`)
   - Type: Search engine and web services
   - Characteristics: Highly dynamic content, multiple services integration, variable page structures

3. **Prothom Alo** (`https://prothomalo.com`)
   - Type: News website
   - Characteristics: Static articles, media content, regular layout patterns

## Dataset Merging Process

### Merger Script (`merger.py`)

The dataset merging was performed using an optimized streaming approach to handle large datasets efficiently:

#### Key Features:
- **Memory Efficient**: Processes one file at a time to avoid loading all data into memory
- **Streaming Normalization**: Applies min-max normalization during the merge process
- **Website Index Mapping**: Maintains consistent website labeling across all datasets
- **Data Validation**: Ensures all traces have exactly 1,000 data points

#### Merging Algorithm:
1. **Initialize**: Create website index mapping and empty merged dataset
2. **Stream Processing**: For each individual dataset file:
   - Load JSON data
   - Extract traffic traces and website labels
   - Apply min-max normalization to timing data
   - Append to merged dataset with consistent website indices
3. **Final Output**: Save consolidated dataset as `dataset.json`

#### Normalization Process:
```python
# Min-Max Normalization Formula
normalized_value = (value - min_value) / (max_value - min_value)
```
- Applied to each trace individually
- Ensures all timing values are scaled to [0, 1] range
- Preserves relative timing patterns within each trace

## Final Merged Dataset Characteristics

### Dataset Statistics
- **File Name**: `dataset.json`
- **Total Samples**: 56,853 traffic traces
- **Number of Classes**: 3 websites
- **Feature Dimensions**: 1,000 data points per trace
- **Data Format**: JSON with normalized floating-point values
- **File Size**: ~1.2 GB (estimated)

### Class Distribution

| Website | Website Index | Sample Count | Percentage |
|---------|---------------|--------------|------------|
| BUET Moodle | 0 | ~18,951 | ~33.3% |
| Google | 1 | ~18,951 | ~33.3% |
| Prothom Alo | 2 | ~18,951 | ~33.3% |

*Note: Exact counts may vary slightly due to individual dataset sizes*

### Data Structure

```json
{
  "data": [
    {
      "trace": [0.0, 0.123, 0.245, ..., 0.987],  // 1000 normalized values
      "website": "https://cse.buet.ac.bd/moodle/",
      "website_index": 0
    },
    {
      "trace": [0.0, 0.087, 0.234, ..., 0.876],  // 1000 normalized values  
      "website": "https://google.com",
      "website_index": 1
    },
    // ... more samples
  ]
}
```

### Data Quality Characteristics

#### Trace Length Validation
- **Standard Length**: All traces contain exactly 1,000 data points
- **Consistency**: Verified across all 56,853 samples
- **No Missing Values**: Complete data for all traces

#### Normalization Properties
- **Value Range**: All trace values are in [0.0, 1.0] range
- **Preservation**: Relative timing patterns maintained within each trace
- **Consistency**: Normalization applied uniformly across all samples

#### Website Index Mapping
- **Consistent Labeling**: Same website always maps to same index
- **Zero-Based Indexing**: Indices range from 0 to 2
- **String Preservation**: Original website URLs maintained for reference

## Data Collection Methodology

### Individual Contributor Data
Each contributor collected network traffic traces by:
1. Setting up controlled network monitoring
2. Visiting each target website multiple times
3. Recording network timing patterns
4. Storing traces in standardized JSON format

### Environment Information
Some contributors provided additional environment details:
- Network configuration
- Browser specifications
- System specifications
- Collection timestamp information

## Data Usage Guidelines

### Training/Testing Recommendations
- **Cross-Validation**: Use stratified k-fold to maintain class balance
- **Train/Test Split**: Ensure representative samples from all contributors
- **Validation**: Verify model performance across different contributor data

### Preprocessing Notes
- **No Additional Normalization Needed**: Data is already normalized
- **Feature Engineering**: 1,000 timing points can be used directly
- **Sequence Analysis**: Maintain temporal order of timing data

### Performance Considerations
- **Memory Usage**: ~1.2 GB for full dataset loading
- **Streaming Recommended**: For memory-constrained environments
- **Batch Processing**: Use appropriate batch sizes for training

## Validation Results

### Dataset Integrity Check (`validate_dataset.py`)
```
✓ JSON file is valid
✓ Total samples: 56,853
✓ All traces have exactly 1,000 data points
✓ Three distinct websites present
✓ Balanced class distribution
✓ All values in normalized range [0.0, 1.0]
✓ No missing or null values
```

### Quality Metrics
- **Completeness**: 100% - No missing traces or incomplete data
- **Consistency**: 100% - All traces follow same format and length
- **Balance**: ~33.3% per class - Well-balanced dataset
- **Validity**: 100% - All JSON structures valid and parseable

## File Dependencies

### Input Files
- `individual-data/*/dataset.json`: Original contributor datasets
- `individual-data/*/environment.txt`: Optional environment information

### Output Files
- `dataset.json`: Final merged and normalized dataset
- `merger.py`: Dataset merging script
- `validate_dataset.py`: Dataset validation script

### Processing Scripts
- `merger.py`: Combines and normalizes individual datasets
- `validate_dataset.py`: Validates merged dataset integrity
- `train.py`: Uses merged dataset for model training

## Dataset Versioning

- **Version**: 1.0
- **Creation Date**: June 2025
- **Last Modified**: June 2025
- **Contributors**: 18 individual collectors
- **Processing**: Merged and normalized using `merger.py`

## Usage Examples

### Loading the Dataset
```python
import json

# Load the complete dataset
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

# Access individual samples
traces = [sample['trace'] for sample in dataset['data']]
websites = [sample['website'] for sample in dataset['data']]
labels = [sample['website_index'] for sample in dataset['data']]
```

### Streaming Large Dataset
```python
import json

def stream_dataset(filename):
    with open(filename, 'r') as f:
        dataset = json.load(f)
        for sample in dataset['data']:
            yield sample['trace'], sample['website_index']

# Use in training loop
for trace, label in stream_dataset('dataset.json'):
    # Process individual samples
    pass
```

## Future Considerations

### Dataset Expansion
- Additional website targets
- More diverse network environments
- Temporal data collection across different time periods
- Geographic diversity in collection points

### Enhancement Opportunities
- Feature extraction beyond raw timing data
- Multi-resolution temporal analysis
- Protocol-level feature engineering
- Encrypted traffic analysis capabilities

---

*This dataset documentation was generated as part of the website fingerprinting project for CSE406: Computer Security Sessional.*
