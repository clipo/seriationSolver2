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

    def plot_sequence(self,
                      assemblages: pd.DataFrame,
                      sequence_indices: List[int],
                      title: str = "Seriation Sequence",
                      save_path: str = None):
        """Create visualizations for a valid sequence"""
        # Create both heatmap and battleship plots
        fig = plt.figure(figsize=(20, 12))

        # Heatmap
        ax1 = plt.subplot(121)
        sequence_data = assemblages.iloc[sequence_indices]
        totals = sequence_data.sum(axis=1)
        proportions = sequence_data.div(totals, axis=0)
        sns.heatmap(proportions, cmap='YlOrRd', annot=True, fmt='.3f', ax=ax1)
        ax1.set_title(f'{title} - Proportions')
        ax1.set_ylabel('Assemblages')
        ax1.set_xlabel('Types')

        # Battleship plots
        ax2 = plt.subplot(122)
        n_types = len(assemblages.columns)
        proportions_array = proportions.values * 100

        for i, type_name in enumerate(assemblages.columns):
            values = proportions_array[:, i]
            baseline = i * max(values.max(), 1)  # Stack the plots

            # Create the battleship shape
            ax2.fill_between(range(len(values)),
                             baseline + values,
                             baseline,
                             alpha=0.6)
            ax2.plot(range(len(values)), baseline + values, 'k-', linewidth=1)

            # Add type name
            ax2.text(-0.5, baseline + max(values) / 2, type_name,
                     ha='right', va='center')

        ax2.set_title(f'{title} - Battleship Curves')
        ax2.set_xlim(-1, len(sequence_indices))
        ax2.set_xticks(range(len(sequence_indices)))
        ax2.set_xticklabels(sequence_data.index, rotation=45, ha='right')

        plt.tight_layout()

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
        for i, sequence in enumerate(sorted_sequences):
            sequence_names = assemblages.index[list(sequence)].tolist()
            results[f'Sequence_{i}'] = sequence_names

            # Create visualizations
            self.plot_sequence(
                assemblages,
                list(sequence),
                f'Seriation Sequence {i}',
                save_path=os.path.join(output_dir, f'sequence_{i}.png')
            )

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
        # Read the data
        print("Loading data...")
        data = pd.read_csv('pfg-cpl.txt', sep='\t', index_col=0)
        print(f"Loaded {len(data)} assemblages with {len(data.columns)} types\n")

        # Initialize solver
        solver = DPSeriationSolver(min_group_size=3)

        # Find and visualize seriation groups
        results = solver.fit(data)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()