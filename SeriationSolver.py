import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.stats import beta
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class SeriationSolver:
    def __init__(self,
                 min_group_size: int = 3,
                 confidence_level: float = 0.95,
                 n_bootstrap: int = 1000):
        self.min_group_size = min_group_size
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()

    def create_training_data(self, assemblages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data by generating valid and invalid sequences"""
        print("Generating training data...")
        X = []
        y = []
        n_assemblages = len(assemblages)
        row_totals = assemblages.sum(axis=1)

        for type_idx in tqdm(range(assemblages.shape[1]), desc="Processing types"):
            for length in range(3, min(n_assemblages + 1, 8)):  # Limit length for efficiency
                for start in range(n_assemblages - length + 1):
                    sequence = assemblages[start:start + length, type_idx]
                    totals = row_totals[start:start + length]
                    props = sequence / totals

                    if np.sum(props) > 0:  # Only use non-zero sequences
                        # Add valid sequence
                        X.append(self._pad_sequence(props))
                        y.append(1)

                        # Add invalid sequences (random permutations)
                        for _ in range(2):
                            perm = np.random.permutation(len(props))
                            X.append(self._pad_sequence(props[perm]))
                            y.append(0)

        return np.array(X), np.array(y)

    def _pad_sequence(self, seq: np.ndarray, max_len: int = 8) -> np.ndarray:
        """Pad sequence to fixed length"""
        padded = np.zeros(max_len)
        padded[:len(seq)] = seq
        return padded

    def train_model(self, assemblages: np.ndarray):
        """Train the RandomForest model"""
        X, y = self.create_training_data(assemblages)
        X_scaled = self.scaler.fit_transform(X)

        print("Training model...")
        self.model.fit(X_scaled, y)
        print(f"Model training complete. Score: {self.model.score(X_scaled, y):.3f}")

    def predict_sequence_validity(self, sequence: np.ndarray) -> float:
        """Predict if a sequence is valid using the trained model"""
        padded = self._pad_sequence(sequence)
        scaled = self.scaler.transform(padded.reshape(1, -1))
        return self.model.predict_proba(scaled)[0][1]

    def bootstrap_test_group(self,
                             assemblages: np.ndarray,
                             group_indices: np.ndarray) -> Dict[str, float]:
        """Perform bootstrap testing on a potential seriation group"""
        n_samples = len(group_indices)
        group_data = assemblages[group_indices]
        row_totals = group_data.sum(axis=1)
        results = {
            'stability_score': 0.0,
            'confidence': 0.0,
            'mean_validity': 0.0
        }

        valid_bootstraps = 0
        validity_scores = []

        for _ in tqdm(range(self.n_bootstrap), desc="Bootstrap testing", leave=False):
            # Bootstrap sample with replacement
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_data = group_data[bootstrap_indices]
            bootstrap_totals = row_totals[bootstrap_indices]

            # Calculate proportions
            bootstrap_props = bootstrap_data / bootstrap_totals[:, np.newaxis]

            # Check validity of each type's sequence
            type_validities = []
            for type_idx in range(assemblages.shape[1]):
                type_props = bootstrap_props[:, type_idx]
                if np.sum(type_props) > 0:  # Only check non-zero sequences
                    validity = self.predict_sequence_validity(type_props)
                    type_validities.append(validity)

            mean_validity = np.mean(type_validities) if type_validities else 0
            validity_scores.append(mean_validity)
            if mean_validity > 0.5:
                valid_bootstraps += 1

        results['stability_score'] = np.std(validity_scores)
        results['confidence'] = valid_bootstraps / self.n_bootstrap
        results['mean_validity'] = np.mean(validity_scores)

        return results

    def estimate_combinations(self, n_assemblages: int) -> int:
        """
        Estimate the number of possible combinations to test.
        Ignores pairs as they're trivial solutions.
        """
        total = 0
        # For each possible group size from min_group_size to n_assemblages
        for size in range(max(3, self.min_group_size), n_assemblages + 1):
            # Number of possible contiguous sequences of this size
            total += n_assemblages - size + 1

        return total

    def find_seriation_groups(self,
                              assemblages: np.ndarray,
                              n_attempts: int = 100,
                              coverage_threshold: float = 0.10,  # Stop if we reach 10% coverage
                              max_unchanged_attempts: int = 1000,  # Stop if no new combinations found
                              min_group_size: int = 3  # Minimum size for valid groups
                              ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Find groups of assemblages that can be seriated together.
        Includes early stopping criteria and ignores pairs.
        """
        n_assemblages = len(assemblages)
        best_groups = []

        # Estimate total combinations possible
        total_possible = self.estimate_combinations(n_assemblages)
        print(f"\nTotal possible combinations: {total_possible:,}")
        print(
            f"Initial test plan: {n_attempts} attempts ({(n_attempts / total_possible * 100):.2f}% of possible combinations)")
        print(f"Will stop early if:")
        print(f"  - Reach {coverage_threshold * 100:.1f}% coverage")
        print(f"  - No new combinations found in {max_unchanged_attempts} attempts")

        # Train the model
        self.train_model(assemblages)

        # Create overall progress bar
        overall_progress = tqdm(total=n_attempts, desc="Overall progress", position=0)
        tested_combinations = set()
        unchanged_counter = 0

        print("\nSearching for seriation groups...")
        for attempt in range(n_attempts):
            current_ordering = np.random.permutation(n_assemblages)
            current_group = []
            any_new = False

            for i in range(n_assemblages):
                temp_group = current_group + [current_ordering[i]]
                if len(temp_group) >= min_group_size:  # Only consider groups of 3 or more
                    # Convert group to sortable tuple for tracking
                    group_key = tuple(sorted(temp_group))

                    # Only test if we haven't seen this combination before
                    if group_key not in tested_combinations:
                        tested_combinations.add(group_key)
                        any_new = True

                        bootstrap_stats = self.bootstrap_test_group(assemblages, np.array(temp_group))

                        if bootstrap_stats['confidence'] > 0.8 and bootstrap_stats['mean_validity'] > 0.6:
                            current_group = temp_group
                        else:
                            if len(current_group) >= min_group_size:
                                stats = self.bootstrap_test_group(assemblages, np.array(current_group))
                                best_groups.append((np.array(current_group), stats))
                            current_group = [current_ordering[i]]
                else:
                    current_group.append(current_ordering[i])

            # Add last group if valid
            if len(current_group) >= min_group_size:
                group_key = tuple(sorted(current_group))
                if group_key not in tested_combinations:
                    tested_combinations.add(group_key)
                    any_new = True
                    stats = self.bootstrap_test_group(assemblages, np.array(current_group))
                    if stats['confidence'] > 0.8 and stats['mean_validity'] > 0.6:
                        best_groups.append((np.array(current_group), stats))

            # Update progress only if we tested new combinations
            if any_new:
                overall_progress.update(1)
                unchanged_counter = 0
            else:
                unchanged_counter += 1

            # Provide periodic updates on unique combinations tested
            current_coverage = len(tested_combinations) / total_possible
            if attempt % 10 == 0:
                print(f"\rUnique combinations tested: {len(tested_combinations):,} "
                      f"({current_coverage * 100:.2f}% of total possible)",
                      end="")

            # Check early stopping criteria
            if current_coverage >= coverage_threshold:
                print(f"\n\nReached {coverage_threshold * 100:.1f}% coverage threshold. Stopping early.")
                break

            if unchanged_counter >= max_unchanged_attempts:
                print(f"\n\nNo new combinations found in {max_unchanged_attempts} attempts. Stopping early.")
                break

        overall_progress.close()
        print(f"\nFinal unique combinations tested: {len(tested_combinations):,} "
              f"({(len(tested_combinations) / total_possible * 100):.2f}% of total possible)")

        # Remove duplicates and sort by confidence
        unique_groups = []
        seen = set()
        for group, stats in sorted(best_groups, key=lambda x: x[1]['confidence'], reverse=True):
            group_key = tuple(sorted(group))
            if group_key not in seen:
                seen.add(group_key)
                unique_groups.append((group, stats))

        return unique_groups

    def plot_seriation_group(self,
                             assemblages: pd.DataFrame,
                             group_indices: np.ndarray,
                             bootstrap_stats: Dict,
                             title: str = "Seriation Group"):
        """Plot seriation group with bootstrap statistics"""
        group_data = assemblages.iloc[group_indices]
        row_totals = group_data.sum(axis=1)
        proportions = group_data.div(row_totals, axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot proportions
        sns.heatmap(proportions, cmap='YlOrRd', annot=True, fmt='.3f', ax=ax1)
        ax1.set_title(f'{title}\nConfidence: {bootstrap_stats["confidence"]:.3f}\n'
                      f'Stability: {1 - bootstrap_stats["stability_score"]:.3f}')
        ax1.set_ylabel('Assemblages')
        ax1.set_xlabel('Types')

        # Plot raw counts
        sns.heatmap(group_data, cmap='YlOrRd', annot=True, fmt='d', ax=ax2)
        ax2.set_title(f'{title} - Raw Counts')
        ax2.set_ylabel('Assemblages')
        ax2.set_xlabel('Types')

        plt.tight_layout()
        plt.show()

    def plot_seriation_group(self,
                             assemblages: pd.DataFrame,
                             group_indices: np.ndarray,
                             bootstrap_stats: Dict,
                             title: str = "Seriation Group",
                             save_path: str = None):
        """Plot seriation group with bootstrap statistics"""
        group_data = assemblages.iloc[group_indices]
        row_totals = group_data.sum(axis=1)
        proportions = group_data.div(row_totals, axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot proportions
        sns.heatmap(proportions, cmap='YlOrRd', annot=True, fmt='.3f', ax=ax1)
        ax1.set_title(f'{title}\nConfidence: {bootstrap_stats["confidence"]:.3f}\n'
                      f'Stability: {1 - bootstrap_stats["stability_score"]:.3f}')
        ax1.set_ylabel('Assemblages')
        ax1.set_xlabel('Types')

        # Plot raw counts
        sns.heatmap(group_data, cmap='YlOrRd', annot=True, fmt='d', ax=ax2)
        ax2.set_title(f'{title} - Raw Counts')
        ax2.set_ylabel('Assemblages')
        ax2.set_xlabel('Types')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_battleship(self,
                        assemblages: pd.DataFrame,
                        group_indices: np.ndarray,
                        title: str = "Battleship Plot",
                        save_path: str = None):
        """Create a traditional battleship plot for a seriation group"""
        group_data = assemblages.iloc[group_indices]
        row_totals = group_data.sum(axis=1)
        proportions = group_data.div(row_totals, axis=0) * 100  # Convert to percentages

        # Create figure with a subplot for each type
        n_types = len(assemblages.columns)
        fig_height = max(8, n_types * 1.5)  # Adjust height based on number of types
        fig, axes = plt.subplots(n_types, 1, figsize=(10, fig_height))
        fig.suptitle(title, fontsize=16, y=0.95)

        # Create a plot for each type
        for i, (type_name, ax) in enumerate(zip(assemblages.columns, axes)):
            values = proportions[type_name]

            # Create the battleship shape
            ax.fill_between(range(len(values)), values, color='navy', alpha=0.6)
            ax.plot(values, color='navy', linewidth=2)

            # Add type name and scale
            ax.set_ylabel(f'{type_name}\n(%)', rotation=0, ha='right', va='center')
            ax.set_xlim(-0.5, len(values) - 0.5)
            ax.set_ylim(0, max(values.max() * 1.1, 1))  # Add 10% padding

            # Add assemblage names at bottom of last subplot
            if i == len(axes) - 1:
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(group_data.index, rotation=45, ha='right')
            else:
                ax.set_xticks([])

            # Add gridlines
            ax.grid(True, linestyle='--', alpha=0.3)

            # Add percentage values
            for j, v in enumerate(values):
                if v > 0:  # Only show non-zero values
                    ax.text(j, v, f'{v:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def fit(self, assemblages: pd.DataFrame, output_dir: str = 'seriation_results') -> Dict[str, Dict]:
        """Main method to find seriation solution"""
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        assemblage_matrix = assemblages.values

        # Find groups with bootstrap statistics
        groups_with_stats = self.find_seriation_groups(assemblage_matrix)

        # Format results and create visualizations
        results = {}

        if groups_with_stats:
            print("\nCreating visualizations...")

            # First create and save heatmaps for each group
            for i, (group, stats) in enumerate(groups_with_stats):
                group_names = assemblages.index[group].tolist()
                results[f'Group_{i}'] = {
                    'assemblages': group_names,
                    'bootstrap_stats': stats
                }

                print(f"\nVisualizing Group {i}...")

                # Save heatmap
                heatmap_path = os.path.join(output_dir, f'heatmap_group_{i}.png')
                self.plot_seriation_group(assemblages, group, stats,
                                          f'Seriation Group {i}',
                                          save_path=heatmap_path)

                # Save battleship plot
                battleship_path = os.path.join(output_dir, f'battleship_group_{i}.png')
                self.plot_battleship(assemblages, group,
                                     f'Battleship Plot - Group {i}',
                                     save_path=battleship_path)

            # Save results to text file
            results_path = os.path.join(output_dir, 'seriation_results.txt')
            with open(results_path, 'w') as f:
                f.write("Seriation Analysis Results\n")
                f.write("=========================\n\n")
                for group_name, info in results.items():
                    f.write(f"\n{group_name}:\n")
                    f.write("Assemblages:\n")
                    for assemblage in info['assemblages']:
                        f.write(f"  {assemblage}\n")
                    f.write("\nBootstrap Statistics:\n")
                    for stat, value in info['bootstrap_stats'].items():
                        f.write(f"  {stat}: {value:.3f}\n")

            print(f"\nResults saved to directory: {output_dir}")
            print("Files created:")
            print("  - Heatmap plots (heatmap_group_X.png)")
            print("  - Battleship plots (battleship_group_X.png)")
            print("  - Results summary (seriation_results.txt)")

        else:
            print("\nNo valid seriation groups found.")

        return results


if __name__ == "__main__":
    print("Starting seriation analysis...")

    try:
        # Read the data
        print("Loading data...")
        data = pd.read_csv('pfg-cpl.txt', sep='\t', index_col=0)
        print(f"Loaded {len(data)} assemblages with {len(data.columns)} types\n")

        # Create output directory
        output_dir = 'seriation_results'

        # Initialize solver
        solver = SeriationSolver(min_group_size=3, n_bootstrap=1000)

        # Find and visualize seriation groups
        results = solver.fit(data, output_dir=output_dir)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()