import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
from scipy.stats import beta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
from itertools import combinations
import concurrent.futures
import os
import argparse


def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load data from a file. Supports CSV, TSV, and Excel files.

    Parameters:
    - file_path: Path to data file. If None, uses default 'pfg-cpl.txt'

    Returns:
    - DataFrame with assemblage data
    """
    default_file = 'pfg-cpl.txt'

    if file_path is None:
        file_path = default_file
        print(f"No file specified. Using default file: {file_path}")
    else:
        print(f"Loading data from: {file_path}")

    try:
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()

        # Load based on file type
        if file_ext == '.xlsx' or file_ext == '.xls':
            data = pd.read_excel(file_path, index_col=0)
            print("Loaded Excel file")
        elif file_ext == '.csv':
            data = pd.read_csv(file_path, index_col=0)
            print("Loaded CSV file")
        elif file_ext == '.txt' or file_ext == '.tsv':
            data = pd.read_csv(file_path, sep='\t', index_col=0)
            print("Loaded tab-delimited file")
        else:
            # Try to infer format
            try:
                data = pd.read_csv(file_path, sep='\t', index_col=0)
                print("Loaded as tab-delimited file")
            except:
                try:
                    data = pd.read_csv(file_path, index_col=0)
                    print("Loaded as CSV file")
                except:
                    try:
                        data = pd.read_excel(file_path, index_col=0)
                        print("Loaded as Excel file")
                    except:
                        raise ValueError("Unable to load file. Please ensure it is CSV, TSV, or Excel format.")

        print(f"Data shape: {data.shape} (assemblages × types)")
        return data

    except Exception as e:
        print(f"Error loading file: {str(e)}")
        raise


class DPSeriationSolver:
    def __init__(self,
                 min_group_size: int = 3,
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 1000):
        self.min_group_size = min_group_size
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.valid_sequences = {}

    def check_monotonicity_statistical(self,
                                       sequence: np.ndarray,
                                       totals: np.ndarray,
                                       tolerance: float = 0.1) -> bool:
        """Check if a sequence follows battleship curve pattern using confidence intervals"""
        if len(sequence) < 3:
            return True

        # Convert to proportions
        proportions = sequence / totals

        # Ignore very small proportions that might be noise
        min_prop = proportions.max() * 0.05
        proportions = np.where(proportions < min_prop, 0, proportions)

        # Find peak
        peak_idx = np.argmax(proportions)

        # Check monotonicity before and after peak with tolerance
        if peak_idx > 0:
            before_peak = proportions[:peak_idx + 1]
            diffs_before = np.diff(before_peak)
            if not np.all(diffs_before >= -tolerance * before_peak[:-1].max()):
                return False

        if peak_idx < len(proportions) - 1:
            after_peak = proportions[peak_idx:]
            diffs_after = np.diff(after_peak)
            if not np.all(diffs_after <= tolerance * after_peak[:-1].max()):
                return False

        return True

    def check_sequence_validity(self,
                                assemblages: np.ndarray,
                                indices: List[int]) -> bool:
        """Check if a sequence of assemblages forms a valid seriation"""
        sequence = assemblages[indices]
        row_totals = sequence.sum(axis=1)

        # Check each type for monotonicity
        for type_idx in range(assemblages.shape[1]):
            type_sequence = sequence[:, type_idx]
            if np.sum(type_sequence) > 0:  # Only check types that appear
                if not self.check_monotonicity_statistical(type_sequence, row_totals):
                    return False
        return True

    def find_valid_subsequence(self,
                               assemblages: np.ndarray,
                               start_idx: int,
                               max_length: int) -> Set[Tuple[int]]:
        """Find valid subsequences starting at a given index"""
        valid_seqs = set()
        n = len(assemblages)

        # Try sequences of increasing length
        for length in range(self.min_group_size, max_length + 1):
            if start_idx + length > n:
                break

            indices = list(range(start_idx, start_idx + length))
            if self.check_sequence_validity(assemblages, indices):
                valid_seqs.add(tuple(indices))

        return valid_seqs

    def process_chunk(self,
                      args: Tuple[np.ndarray, int, int]) -> Set[Tuple[int]]:
        """Process a chunk of starting positions (for parallel processing)"""
        assemblages, start_idx, max_length = args
        return self.find_valid_subsequence(assemblages, start_idx, max_length)

    def find_all_valid_sequences(self,
                                 assemblages: np.ndarray,
                                 max_cores: int = None) -> Set[Tuple[int]]:
        """
        Find all valid seriation sequences using parallel dynamic programming
        """
        n = len(assemblages)
        if max_cores is None:
            max_cores = mp.cpu_count()

        # Create chunks for parallel processing
        chunks = [(assemblages, i, n - i) for i in range(n - self.min_group_size + 1)]

        valid_sequences = set()

        # Process chunks in parallel
        print(f"\nSearching for valid sequences using {max_cores} cores...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]

            # Show progress bar
            with tqdm(total=len(futures), desc="Processing subsequences") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    valid_sequences.update(future.result())
                    pbar.update(1)

        print(f"Found {len(valid_sequences)} valid sequences")
        return valid_sequences

        def plot_battleship(self,
                            assemblages: pd.DataFrame,
                            sequence_indices: List[int],
                            title: str = "Seriation Sequence",
                            save_path: str = None):
            """Create traditional battleship plot with centered bars varying in width"""
            sequence_data = assemblages.iloc[sequence_indices]
            totals = sequence_data.sum(axis=1)
            proportions = sequence_data.div(totals, axis=0) * 100  # Convert to percentages

            # Create figure
            n_types = len(assemblages.columns)
            fig_width = max(12, n_types * 2)  # Adjust width based on number of types
            fig_height = max(8, len(sequence_indices) * 0.4)  # Adjust height based on assemblages
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Get max proportion for scaling
            max_prop = proportions.max().max()
            bar_height = 0.6  # Fixed height for bars
            type_positions = np.arange(n_types) * 3  # Triple spacing between types

            # Create plot
            ax = plt.gca()

            # Plot bars
            for i, (_, row) in enumerate(sequence_data.iterrows()):
                row_props = proportions.iloc[i]

                # Plot bars for each type
                for j, (type_name, prop) in enumerate(row_props.items()):
                    if prop > 0:  # Only plot non-zero values
                        # Width varies with proportion, height is fixed
                        width = prop / max_prop  # Scale width by proportion
                        x_pos = type_positions[j]
                        ax.bar(x=x_pos,
                               height=bar_height,
                               width=width,
                               bottom=i - bar_height / 2,
                               color='black',
                               alpha=0.7)

            # Customize plot
            ax.set_ylim(-0.5, len(sequence_indices) - 0.5)
            ax.set_xlim(min(type_positions) - 1.5, max(type_positions) + 1.5)

            # Add labels
            ax.set_yticks(range(len(sequence_indices)))
            ax.set_yticklabels(sequence_data.index)

            # Add type names at bottom
            ax.set_xticks(type_positions)
            ax.set_xticklabels(assemblages.columns, rotation=45, ha='right')

            # Remove unnecessary spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add title
            plt.title(title)
            plt.tight_layout()

            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        def plot_heatmap(self,
                         assemblages: pd.DataFrame,
                         sequence_indices: List[int],
                         title: str = "Heatmap Sequence",
                         save_path: str = None):
            """Create heatmap visualization"""
            plt.figure(figsize=(12, 8))

            sequence_data = assemblages.iloc[sequence_indices]
            totals = sequence_data.sum(axis=1)
            proportions = sequence_data.div(totals, axis=0)

            sns.heatmap(proportions, cmap='YlOrRd', annot=True, fmt='.3f')
            plt.title(title)
            plt.ylabel('Assemblages')
            plt.xlabel('Types')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

        def fit(self,
                assemblages: pd.DataFrame,
                output_dir: str = 'seriation_results',
                max_cores: int = None) -> Dict[str, List[str]]:
            """
            Main method to find seriation solutions using parallel dynamic programming
            """
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            print("Starting parallel dynamic programming seriation analysis...")
            print(f"Data: {len(assemblages)} assemblages, {len(assemblages.columns)} types")

            # Find all valid sequences
            valid_sequences = self.find_all_valid_sequences(assemblages.values, max_cores)

            # Sort sequences by length (longest first)
            sorted_sequences = sorted(valid_sequences,
                                      key=len,
                                      reverse=True)

            # Prepare results
            results = {}
            sequence_list = []

            for i, sequence in enumerate(sorted_sequences):
                sequence_names = assemblages.index[list(sequence)].tolist()
                results[f'Sequence_{i}'] = sequence_names
                sequence_list.append(list(sequence))

                # Create visualizations
                # Save battleship plot
                battleship_path = os.path.join(output_dir, f'battleship_{i}.png')
                self.plot_battleship(
                    assemblages,
                    list(sequence),
                    f'Seriation Sequence {i}',
                    save_path=battleship_path
                )
                print(f"Saved battleship plot to: {battleship_path}")

                # Save heatmap separately
                heatmap_path = os.path.join(output_dir, f'heatmap_{i}.png')
                self.plot_heatmap(
                    assemblages,
                    list(sequence),
                    f'Heatmap Sequence {i}',
                    save_path=heatmap_path
                )
                print(f"Saved heatmap to: {heatmap_path}")

            # Create combined visualization
            print("\nCreating combined visualization...")
            combined_path = os.path.join(output_dir, 'all_sequences.png')

            try:
                # Create figure for combined plot
                n_sequences = len(sequence_list)
                n_types = len(assemblages.columns)

                if n_sequences > 0:
                    # Make figure size proportional to number of sequences
                    fig = plt.figure(figsize=(15, 4 * n_sequences))
                    plt.suptitle("All Valid Seriation Sequences", y=1.02, fontsize=14)

                    # Create subplots for each sequence
                    for idx, sequence in enumerate(sequence_list):
                        print(f"Adding sequence {idx + 1} to combined plot...")

                        # Create subplot
                        ax = plt.subplot(n_sequences, 1, idx + 1)

                        sequence_data = assemblages.iloc[sequence]
                        totals = sequence_data.sum(axis=1)
                        proportions = sequence_data.div(totals, axis=0) * 100

                        # Plot bars
                        max_prop = proportions.max().max()
                        bar_height = 0.6
                        type_positions = np.arange(n_types) * 3

                        for i, (_, row) in enumerate(sequence_data.iterrows()):
                            row_props = proportions.iloc[i]
                            for j, (type_name, prop) in enumerate(row_props.items()):
                                if prop > 0:
                                    width = 2 * (prop / max_prop)
                                    x_pos = type_positions[j]
                                    ax.bar(x=x_pos,
                                           height=bar_height,
                                           width=width,
                                           bottom=i - bar_height / 2,
                                           color='black',
                                           alpha=0.7)

                        # Customize subplot
                        ax.set_ylim(-0.5, len(sequence) - 0.5)
                        ax.set_xlim(min(type_positions) - 1.5, max(type_positions) + 1.5)

                        # Add labels
                        ax.set_yticks(range(len(sequence)))
                        ax.set_yticklabels(sequence_data.index)

                        ax.set_xticks(type_positions)
                        if idx == len(sequence_list) - 1:
                            ax.set_xticklabels(assemblages.columns, rotation=45, ha='right')
                        else:
                            ax.set_xticklabels([])

                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.set_title(f'Sequence {idx + 1}', pad=10)

                    plt.tight_layout()
                    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    print(f"Saved combined plot to: {combined_path}")

                    # Verify file exists
                    if os.path.exists(combined_path):
                        print(f"Verified: Combined plot file exists at {combined_path}")
                        print(f"File size: {os.path.getsize(combined_path)} bytes")
                    else:
                        print("Warning: Combined plot file was not created successfully")

            except Exception as e:
                print(f"Error creating combined visualization: {str(e)}")
                import traceback
                traceback.print_exc()

            # Save results to text file
            results_path = os.path.join(output_dir, 'seriation_results.txt')
            with open(results_path, 'w') as f:
                f.write("Seriation Analysis Results\n")
                f.write("=========================\n\n")
                for seq_name, assemblage_list in results.items():
                    f.write(f"\n{seq_name}:\n")
                    f.write("Assemblages:\n")
                    for assemblage in assemblage_list:
                        f.write(f"  {assemblage}\n")
                    f.write(f"Length: {len(assemblage_list)}\n")

            print(f"\nResults saved to directory: {output_dir}")
            print(f"Found {len(results)} valid sequences")
            return results

    if __name__ == "__main__":
        print("Starting seriation analysis...")

        try:
            # Set up argument parsing
            parser = argparse.ArgumentParser(description='Run frequency seriation analysis.')
            parser.add_argument('--file', type=str, help='Path to input data file (CSV, TSV, or Excel)', default=None)
            parser.add_argument('--min_group', type=int, help='Minimum group size (default: 3)', default=3)
            parser.add_argument('--confidence', type=float, help='Confidence level (default: 0.95)', default=0.95)
            parser.add_argument('--bootstrap', type=int, help='Number of bootstrap iterations (default: 1000)',
                                default=1000)
            parser.add_argument('--output', type=str, help='Output directory (default: seriation_results)',
                                default='seriation_results')
            parser.add_argument('--cores', type=int, help='Number of CPU cores to use (default: all)', default=None)

            args = parser.parse_args()

            # Load data
            print("Loading data...")
            data = load_data(args.file)

            # Initialize solver with command line parameters
            solver = DPSeriationSolver(
                min_group_size=args.min_group,
                confidence_level=args.confidence,
                n_bootstrap=args.bootstrap
            )

            # Find and visualize seriation groups
            results = solver.fit(
                data,
                output_dir=args.output,
                max_cores=args.cores
            )

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_battleship(self,
                        assemblages: pd.DataFrame,
                        sequence_indices: List[int],
                        title: str = "Seriation Sequence",
                        save_path: str = None):
        """Create traditional battleship plot with centered bars varying in width"""
        sequence_data = assemblages.iloc[sequence_indices]
        totals = sequence_data.sum(axis=1)
        proportions = sequence_data.div(totals, axis=0) * 100  # Convert to percentages

        # Create figure
        n_types = len(assemblages.columns)
        fig_width = max(12, n_types * 2)  # Adjust width based on number of types
        fig_height = max(8, len(sequence_indices) * 0.4)  # Adjust height based on assemblages
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Get max proportion for scaling
        max_prop = proportions.max().max()
        bar_height = 0.6  # Fixed height for bars
        type_positions = np.arange(n_types) * 3  # Triple spacing between types

        # Create plot
        ax = plt.gca()

        # Plot bars
        for i, (_, row) in enumerate(sequence_data.iterrows()):
            row_props = proportions.iloc[i]

            # Plot bars for each type
            for j, (type_name, prop) in enumerate(row_props.items()):
                if prop > 0:  # Only plot non-zero values
                    # Width varies with proportion, height is fixed
                    width = prop / max_prop  # Scale width by proportion
                    x_pos = type_positions[j]
                    ax.bar(x=x_pos,
                           height=bar_height,
                           width=width,
                           bottom=i - bar_height / 2,
                           color='black',
                           alpha=0.7)

        # Customize plot
        ax.set_ylim(-0.5, len(sequence_indices) - 0.5)
        ax.set_xlim(min(type_positions) - 1.5, max(type_positions) + 1.5)

        # Add labels
        ax.set_yticks(range(len(sequence_indices)))
        ax.set_yticklabels(sequence_data.index)

        # Add type names at bottom
        ax.set_xticks(type_positions)
        ax.set_xticklabels(assemblages.columns, rotation=45, ha='right')

        # Remove unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add title
        plt.title(title)
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_heatmap(self,
                     assemblages: pd.DataFrame,
                     sequence_indices: List[int],
                     title: str = "Heatmap Sequence",
                     save_path: str = None):
        """Create heatmap visualization"""
        plt.figure(figsize=(12, 8))

        sequence_data = assemblages.iloc[sequence_indices]
        totals = sequence_data.sum(axis=1)
        proportions = sequence_data.div(totals, axis=0)

        sns.heatmap(proportions, cmap='YlOrRd', annot=True, fmt='.3f')
        plt.title(title)
        plt.ylabel('Assemblages')
        plt.xlabel('Types')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def fit(self,
            assemblages: pd.DataFrame,
            output_dir: str = 'seriation_results',
            max_cores: int = None) -> Dict[str, List[str]]:
        """
        Main method to find seriation solutions using parallel dynamic programming
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("Starting parallel dynamic programming seriation analysis...")
        print(f"Data: {len(assemblages)} assemblages, {len(assemblages.columns)} types")

        # Find all valid sequences
        valid_sequences = self.find_all_valid_sequences(assemblages.values, max_cores)

        # Sort sequences by length (longest first)
        sorted_sequences = sorted(valid_sequences,
                                  key=len,
                                  reverse=True)

        # Prepare results
        results = {}
        sequence_list = []

        for i, sequence in enumerate(sorted_sequences):
            sequence_names = assemblages.index[list(sequence)].tolist()
            results[f'Sequence_{i}'] = sequence_names
            sequence_list.append(list(sequence))

            # Create visualizations
            # Save battleship plot
            battleship_path = os.path.join(output_dir, f'battleship_{i}.png')
            self.plot_battleship(
                assemblages,
                list(sequence),
                f'Seriation Sequence {i}',
                save_path=battleship_path
            )
            print(f"Saved battleship plot to: {battleship_path}")

            # Save heatmap separately
            heatmap_path = os.path.join(output_dir, f'heatmap_{i}.png')
            self.plot_heatmap(
                assemblages,
                list(sequence),
                f'Heatmap Sequence {i}',
                save_path=heatmap_path
            )
            print(f"Saved heatmap to: {heatmap_path}")

        # Create combined visualization
        print("\nCreating combined visualization...")
        combined_path = os.path.join(output_dir, 'all_sequences.png')

        try:
            # Create figure for combined plot
            n_sequences = len(sequence_list)
            n_types = len(assemblages.columns)

            if n_sequences > 0:
                # Make figure size proportional to number of sequences
                fig = plt.figure(figsize=(15, 4 * n_sequences))
                plt.suptitle("All Valid Seriation Sequences", y=1.02, fontsize=14)

                # Create subplots for each sequence
                for idx, sequence in enumerate(sequence_list):
                    print(f"Adding sequence {idx + 1} to combined plot...")

                    # Create subplot
                    ax = plt.subplot(n_sequences, 1, idx + 1)

                    sequence_data = assemblages.iloc[sequence]
                    totals = sequence_data.sum(axis=1)
                    proportions = sequence_data.div(totals, axis=0) * 100

                    # Plot bars
                    max_prop = proportions.max().max()
                    bar_height = 0.6
                    type_positions = np.arange(n_types) * 3

                    for i, (_, row) in enumerate(sequence_data.iterrows()):
                        row_props = proportions.iloc[i]
                        for j, (type_name, prop) in enumerate(row_props.items()):
                            if prop > 0:
                                width = 2 * (prop / max_prop)
                                x_pos = type_positions[j]
                                ax.bar(x=x_pos,
                                       height=bar_height,
                                       width=width,
                                       bottom=i - bar_height / 2,
                                       color='black',
                                       alpha=0.7)

                    # Customize subplot
                    ax.set_ylim(-0.5, len(sequence) - 0.5)
                    ax.set_xlim(min(type_positions) - 1.5, max(type_positions) + 1.5)

                    # Add labels
                    ax.set_yticks(range(len(sequence)))
                    ax.set_yticklabels(sequence_data.index)

                    ax.set_xticks(type_positions)
                    if idx == len(sequence_list) - 1:
                        ax.set_xticklabels(assemblages.columns, rotation=45, ha='right')
                    else:
                        ax.set_xticklabels([])

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_title(f'Sequence {idx + 1}', pad=10)

                plt.tight_layout()
                plt.savefig(combined_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved combined plot to: {combined_path}")

                # Verify file exists
                if os.path.exists(combined_path):
                    print(f"Verified: Combined plot file exists at {combined_path}")
                    print(f"File size: {os.path.getsize(combined_path)} bytes")
                else:
                    print("Warning: Combined plot file was not created successfully")

        except Exception as e:
            print(f"Error creating combined visualization: {str(e)}")
            import traceback
            traceback.print_exc()

        # Save results to text file
        results_path = os.path.join(output_dir, 'seriation_results.txt')
        with open(results_path, 'w') as f:
            f.write("Seriation Analysis Results\n")
            f.write("=========================\n\n")
            for seq_name, assemblage_list in results.items():
                f.write(f"\n{seq_name}:\n")
                f.write("Assemblages:\n")
                for assemblage in assemblage_list:
                    f.write(f"  {assemblage}\n")
                f.write(f"Length: {len(assemblage_list)}\n")

        print(f"\nResults saved to directory: {output_dir}")
        print(f"Found {len(results)} valid sequences")
        return results


if __name__ == "__main__":
    print("Starting seriation analysis...")

    try:
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Run frequency seriation analysis.')
        parser.add_argument('--file', type=str, help='Path to input data file (CSV, TSV, or Excel)', default=None)
        parser.add_argument('--min_group', type=int, help='Minimum group size (default: 3)', default=3)
        parser.add_argument('--confidence', type=float, help='Confidence level (default: 0.95)', default=0.95)
        parser.add_argument('--bootstrap', type=int, help='Number of bootstrap iterations (default: 1000)',
                            default=1000)
        parser.add_argument('--output', type=str, help='Output directory (default: seriation_results)',
                            default='seriation_results')
        parser.add_argument('--cores', type=int, help='Number of CPU cores to use (default: all)', default=None)
        parser.add_argument('--no-network', action='store_true', help='Skip network analysis', default=False)

        args = parser.parse_args()

        # Load data
        print("Loading data...")
        data = load_data(args.file)

        # Initialize solver with command line parameters
        solver = DPSeriationSolver(
            min_group_size=args.min_group,
            confidence_level=args.confidence,
            n_bootstrap=args.bootstrap
        )

        # Find and visualize seriation groups
        results = solver.fit(
            data,
            output_dir=args.output,
            max_cores=args.cores
        )

        # Run network analysis if not disabled
        if not args.no_network:
            try:
                print("\nRunning network analysis...")
                import network

                results_file = os.path.join(args.output, 'seriation_results.txt')
                if os.path.exists(results_file):
                    G = network.create_seriation_network(results_file, args.output)
                    print("Network analysis complete.")
                else:
                    print("Warning: Could not find seriation_results.txt for network analysis")
            except ImportError:
                print("Warning: Could not import network.py. Network analysis skipped.")
            except Exception as e:
                print(f"Warning: Error during network analysis: {str(e)}")
                print("Network analysis skipped.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()