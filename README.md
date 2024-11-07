# Archaeological Seriation Analysis Tools

A comprehensive Python toolkit for archaeological seriation analysis, supporting both frequency and occurrence seriation methods. This package includes tools for parallel dynamic programming frequency seriation, occurrence seriation, and network analysis of seriation relationships.

![seriations ](seriation_results/all_sequences.png "Seriations")
![seriation network](seriation_results/seriation_network.png "Seriation Solution Network")

## Features

### Frequency Seriation (SeriationSolverDynamic.py)
- Parallel processing for efficient sequence discovery
- Statistical evaluation of monotonicity using confidence intervals
- Multiple visualization outputs:
  - Individual battleship plots for each sequence
  - Heatmap visualizations
  - Combined summary visualization of all valid sequences
- Network analysis of seriation relationships based on shared assemblages
- Progress monitoring and detailed reporting

### Occurrence Seriation (occurrenceSeriation.py)
- Analysis of presence/absence patterns
- Dynamic programming approach for finding largest valid groups
- Individual and combined solution visualizations
- Network analysis of solution relationships
- Command-line interface for easy data input

### Network Analysis (network.py)
- Visualization of relationships between seriation sequences
- Analysis of shared assemblages between sequences
- Network statistics and metrics
- High-resolution output graphics

## Installation

### Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm networkx
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
- networkx

## Usage

### Frequency Seriation

```python
from seriation_solver import DPSeriationSolver

# Read your data
data = pd.read_csv('your_data.txt', sep='\t', index_col=0)

# Initialize solver
solver = DPSeriationSolver(min_group_size=3)

# Find and visualize seriation sequences
results = solver.fit(data)
```

### Occurrence Seriation

```bash
# Run with default test data
python occurrenceSeriation.py

# Run with custom data file
python occurrenceSeriation.py --file path/to/your/data.csv
```

### Network Analysis

```python
from network import create_seriation_network

# Path to seriation results
results_file = "seriation_results/seriation_results.txt"

# Create the network
G = create_seriation_network(results_file)
```

## Input Data Formats

### Frequency Seriation Data
Tab-separated file with:
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

### Occurrence Seriation Data
CSV file with:
- First column: Artifact identifiers ('Ahu' in test data)
- Remaining columns: Presence (1) or absence (0) for each type
- Header row with type names

Example data available in the testdata directory:
```
Ahu,Type1,Type2,Type3
Artifact1,1,0,1
Artifact2,1,1,0
Artifact3,0,1,1
```

## Output Files

The analysis creates a directory (default: 'seriation_results') containing:

### Frequency Seriation Output
- Individual battleship plots (battleship_0.png, battleship_1.png, etc.)
- Heatmap visualizations (heatmap_0.png, heatmap_1.png, etc.)
- Combined visualization of all sequences (all_sequences.png)
- Text summary of results (seriation_results.txt)
- Network visualization (seriation_network.png)

### Occurrence Seriation Output
- Individual solution visualizations (valid_solution_1.png, etc.)
- Combined solution visualization (all_valid_solutions.png)
- Solution network graph (solution_network_graph.png)

## Parameters

### DPSeriationSolver
- `min_group_size`: Minimum number of assemblages in a valid sequence (default: 3)
- `confidence_level`: Confidence level for statistical testing (default: 0.95)
- `n_bootstrap`: Number of bootstrap iterations (default: 1000)

### Occurrence Seriation
- Command line arguments:
  - `--file`: Path to input data file (optional, uses test data if not specified)

## Network Analysis Details

Both frequency and occurrence seriation results can be analyzed using network visualization:

### Frequency Seriation Network
- Nodes represent seriation sequences
- Edges indicate shared assemblages between sequences
- Edge width proportional to number of shared assemblages
- Node size indicates number of assemblages in sequence

### Occurrence Seriation Network
- Nodes represent valid solutions
- Edges indicate shared artifacts between solutions
- Edge labels show specific shared artifacts
- Network layout optimized for clarity

## Example Usage

### Complete Frequency Seriation Analysis
```python
# Example with custom parameters
solver = DPSeriationSolver(
    min_group_size=4,
    confidence_level=0.99,
    n_bootstrap=2000
)

# Run analysis
results = solver.fit(
    assemblages=data,
    output_dir='my_results',
    max_cores=4
)
```

### Occurrence Seriation with Custom Data
```bash
python occurrenceSeriation.py --file my_data.csv
```

## Notes

- Both methods use parallel processing for efficient computation
- Statistical evaluation ensures robust sequence identification
- All visualizations are saved in high resolution (300 DPI)
- Detailed progress information and error messages are displayed
- The package includes sample data for testing and demonstration
- Network analysis can be performed on results from either method

## Directory Structure
```
.
├── SeriationSolverDynamic.py  # Frequency seriation implementation
├── occurrenceSeriation.py     # Occurrence seriation implementation
├── network.py                 # Network analysis tools
├── testdata/                  # Sample data directory
│   └── ahu.csv               # Test data for occurrence seriation
└── README.md                 # This documentation
```
