# Parallel Dynamic Programming Seriation Solver

A Python implementation of frequency seriation analysis using parallel dynamic programming. This tool finds valid seriation sequences in archaeological assemblage data by analyzing frequency patterns and ensuring monotonic distributions ("battleship curves").

## Features

- Parallel processing for efficient sequence discovery
- Statistical evaluation of monotonicity using confidence intervals
- Multiple visualization outputs:
  - Individual battleship plots for each sequence
  - Heatmap visualizations
  - Combined summary visualization of all valid sequences
- Progress monitoring and detailed reporting

## Installation

### Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm
```

### Dependencies

- Python 3.7+
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Usage

### Basic Usage

```python
from seriation_solver import DPSeriationSolver

# Read your data
data = pd.read_csv('your_data.txt', sep='\t', index_col=0)

# Initialize solver
solver = DPSeriationSolver(min_group_size=3)

# Find and visualize seriation sequences
results = solver.fit(data)
```

### Input Data Format

The input data should be a tab-separated file with:
- First column: Assemblage IDs (used as index)
- Remaining columns: Frequencies for each type
- Header row with type names

Example:
```
Assemblage   Type1   Type2   Type3
Sample1      10      5       2
Sample2      8       7       3
Sample3      4       9       5
```

### Output

The solver creates a directory (default: 'seriation_results') containing:
- Individual battleship plots (battleship_0.png, battleship_1.png, etc.)
- Heatmap visualizations (heatmap_0.png, heatmap_1.png, etc.)
- Combined visualization of all sequences (all_sequences.png)
- Text summary of results (seriation_results.txt)

## Parameters

### DPSeriationSolver

- `min_group_size`: Minimum number of assemblages in a valid sequence (default: 3)
- `confidence_level`: Confidence level for statistical testing (default: 0.95)
- `n_bootstrap`: Number of bootstrap iterations (default: 1000)

### fit method

- `assemblages`: Pandas DataFrame containing the assemblage data
- `output_dir`: Directory for saving results (default: 'seriation_results')
- `max_cores`: Maximum number of CPU cores to use (default: all available)

## Output Details

### Battleship Plots
- Vertical bars with width proportional to type frequency
- Assemblages ordered by seriation sequence
- Clear visualization of battleship-curve patterns

### Heatmaps
- Color-coded visualization of frequencies
- Normalized proportions
- Row and column labels

### Combined Visualization
- All valid sequences in one figure
- Consistent scaling across sequences
- Clear labeling and sequence numbering

## Example

```python
# Example with custom parameters
solver = DPSeriationSolver(
    min_group_size=4,
    confidence_level=0.99,
    n_bootstrap=2000
)

# Run analysis with custom output directory
results = solver.fit(
    assemblages=data,
    output_dir='my_results',
    max_cores=4
)
```

## Notes

- The solver uses parallel processing to efficiently search for valid sequences
- Statistical evaluation ensures robust sequence identification
- Visualizations are saved in high resolution (300 DPI)
- Progress information and error messages are displayed during processing