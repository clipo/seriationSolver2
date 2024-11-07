# Archaeological Seriation Analysis Tools v1.0.0

A comprehensive Python toolkit for archaeological seriation analysis, implementing frequency seriation with parallel dynamic programming, occurrence seriation, and network analysis of seriation relationships. Version 1.0.0 introduces maximal sequence filtering and dual network visualizations.

## Version 1.0.0 Features

- Parallel dynamic programming approach for frequency seriation
- Maximal sequence filtering (eliminates subsequences contained in larger sequences)
- Dual network visualization system:
  - Shared assemblage networks
  - Sequential relationship networks with spring layout
- Support for multiple input formats (CSV, TSV, Excel)
- Statistical evaluation of monotonicity
- Comprehensive visualization suite
- Parallel processing capabilities

![freqency seriation](seriation_results/all_sequences.png "Frequency Seriation")
![seriation network](seriation_results/seriation_network.png "Seriation Network")

## Features

### Frequency Seriation (SeriationSolverDynamic.py)
- Parallel processing for efficient sequence discovery
- Statistical evaluation of monotonicity using confidence intervals
- Multiple visualization outputs:
  - Individual battleship plots for each sequence
  - Heatmap visualizations
  - Combined summary visualization of all valid sequences
- Network analysis of seriation relationships with two visualization types:
  - Shared assemblage networks
  - Sequential relationship networks
- Progress monitoring and detailed reporting
- Support for multiple input file formats (CSV, TSV, Excel)

### Occurrence Seriation (occurrenceSeriation.py)
- Analysis of presence/absence patterns
- Dynamic programming approach for finding largest valid groups
- Individual and combined solution visualizations
- Network analysis of solution relationships
- Command-line interface for easy data input

### Network Analysis (network.py)
Two complementary network visualizations:
1. Shared Assemblage Network:
   - Nodes represent seriation sequences
   - Edges show shared assemblages between sequences
   - Edge thickness indicates number of shared assemblages
   - Node size reflects sequence length

2. Sequential Relationship Network:
   - Nodes represent individual assemblages
   - Edges show sequential relationships from all sequences
   - Edge thickness shows how many sequences contain that relationship
   - Spring layout for optimal visualization of relationships
   - Edge labels indicate source sequences
   - Arrows show direction of relationships

## Installation

### Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm networkx openpyxl
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
- openpyxl (for Excel file support)

## Usage

### Frequency Seriation with Network Analysis

```bash
# Basic usage with default settings (includes both network visualizations)
python SeriationSolverDynamic.py

# Specify custom input file
python SeriationSolverDynamic.py --file your_data.csv

# Full options
python SeriationSolverDynamic.py --file your_data.xlsx \
                                --min_group 4 \
                                --confidence 0.99 \
                                --bootstrap 2000 \
                                --output custom_results \
                                --cores 4

# Run without network analysis
python SeriationSolverDynamic.py --file your_data.csv --no-network
```

### Command Line Arguments

- `--file`: Input data file path (CSV, TSV, or Excel)
- `--min_group`: Minimum group size (default: 3)
- `--confidence`: Confidence level (default: 0.95)
- `--bootstrap`: Number of bootstrap iterations (default: 1000)
- `--output`: Output directory (default: 'seriation_results')
- `--cores`: Number of CPU cores to use (default: all available)
- `--no-network`: Skip network analysis (default: False)

### Network Analysis Only

```bash
# Run network analysis on existing results
python network.py
```

## Output Files

The analysis creates a directory (default: 'seriation_results') containing:

### Frequency Seriation Output
- Individual battleship plots (battleship_0.png, battleship_1.png, etc.)
- Heatmap visualizations (heatmap_0.png, heatmap_1.png, etc.)
- Combined visualization of all sequences (all_sequences.png)
- Text summary of results (seriation_results.txt)
- Shared assemblage network visualization (seriation_network.png)
- Sequential relationship network visualization (sequence_network.png)

### Network Visualizations
1. seriation_network.png:
   - Shows relationships between sequences
   - Edge width based on number of shared assemblages
   - Node size based on sequence length
   
2. sequence_network.png:
   - Shows all sequential relationships from all sequences
   - Each assemblage appears once as a node
   - Edge width shows frequency of relationship across sequences
   - Arrows indicate sequence direction
   - Labels show which sequences contain each relationship

## Directory Structure
```
.
├── SeriationSolverDynamic.py  # Main frequency seriation implementation
├── occurrenceSeriation.py     # Occurrence seriation implementation
├── network.py                 # Network analysis and visualization tools
├── testdata/                  # Sample data directory
│   └── ahu.csv               # Test data for occurrence seriation
└── README.md                 # This documentation
```

## Network Analysis Details

### Shared Assemblage Network
- Shows how sequences are related through common assemblages
- Useful for identifying overlapping sequence groups
- Helps validate sequence relationships

### Sequential Relationship Network
- Combines all sequence information into a single graph
- Shows how assemblages are connected across all sequences
- Edge weights show strength of sequential relationships
- Spring layout optimizes visualization of relationships
- Parameters tuned for clarity:
  - Node spacing (`k=1.5`)
  - Layout iterations (500)
  - Edge weight influence
  - Arrow and label positioning

## Notes

- Both seriation methods use parallel processing for efficient computation
- Statistical evaluation ensures robust sequence identification
- Network analysis runs automatically unless disabled
- All visualizations are saved in high resolution (300 DPI)
- The package includes sample data for testing
- Spring layout provides optimal visualization of relationships

## Common Issues and Solutions

1. Network Visualization:
   - If graph is too dense, adjust spring layout parameters
   - Label overlapping can be adjusted through font size
   - Edge weights can be scaled for better visibility

2. File Management:
   - Check output directory permissions
   - Verify result files exist before network analysis
   - Ensure consistent file naming

## References

Key references for methodology:
- Dunnell, R. C. (1970). Seriation Method and Its Evaluation. American Antiquity, 35(3), 305-319.
- O'Brien, M. J., & Lyman, R. L. (1999). Seriation, Stratigraphy, and Index Fossils: The Backbone of Archaeological Dating.
- Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press (for network analysis methods).
