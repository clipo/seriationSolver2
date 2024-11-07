import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from tqdm import tqdm

# Load data
file_path = 'ahu.csv'
artifact_data = pd.read_csv(file_path)


# Function to group artifacts by identical patterns and label them with artifact names
def group_and_label_patterns(df):
    pattern_groups = df.groupby(list(df.columns[1:])).apply(lambda x: ', '.join(x['Ahu'])).reset_index()
    pattern_groups.columns = list(df.columns[1:]) + ['Artifacts']
    pattern_groups[df.columns[1:]] = pattern_groups[df.columns[1:]].astype(int)
    return pattern_groups


# Extract class names for x-axis labels
class_names = artifact_data.columns[1:]
pattern_groups = group_and_label_patterns(artifact_data)


def strict_continuity_check(sorted_patterns):
    return all(np.all(np.diff(np.where(col == 1)[0]) == 1) for col in sorted_patterns.T)


# Dynamic Programming with Memoization for largest valid group solutions
def dp_largest_valid_groups(pattern_df):
    patterns = pattern_df.drop(columns=["Artifacts"]).to_numpy()
    n_patterns = len(patterns)
    memo = {}

    def find_largest_group_ending_at(idx):
        if idx in memo:
            return memo[idx]
        best_group = (idx, idx + 1)
        for start in range(idx - 1, -1, -1):
            candidate_group = patterns[start:idx + 1]
            if len(candidate_group) >= 3 and strict_continuity_check(candidate_group):
                best_group = (start, idx + 1)
            elif len(candidate_group) < 3:
                continue
            else:
                break
        memo[idx] = best_group
        return best_group

    valid_groups = [find_largest_group_ending_at(i) for i in
                    tqdm(range(n_patterns), desc="Finding Largest Valid Groups")]
    unique_groups = list({group for group in valid_groups if group[1] - group[0] >= 3})
    unique_groups.sort(key=lambda x: x[1] - x[0], reverse=True)

    return unique_groups


output_dir = "occurrence_output"
os.makedirs(output_dir, exist_ok=True)


def save_individual_solutions_v2(valid_groups, pattern_df):
    solution_files = []
    for idx, (start, end) in enumerate(tqdm(valid_groups, desc="Saving Individual Solutions")):
        sorted_patterns = pattern_df.iloc[start:end].drop(columns=["Artifacts"]).astype(int).to_numpy()
        labels = pattern_df["Artifacts"].values[start:end]

        plt.figure(figsize=(6, 4))
        plt.imshow(sorted_patterns, cmap="binary", aspect="auto")
        plt.yticks(range(len(labels)), labels, fontsize=8)
        plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
        plt.xlabel("Classes")
        plt.ylabel("Artifact Patterns")
        plt.title(f"Valid Solution {idx + 1}")

        file_path = os.path.join(output_dir, f"valid_solution_{idx + 1}.png")
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        solution_files.append(file_path)

    return solution_files


def plot_all_solutions_v2(solution_files):
    n_solutions = len(solution_files)
    n_cols = 3
    n_rows = (n_solutions + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    for i, file_path in enumerate(solution_files):
        img = plt.imread(file_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Solution {i + 1}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    combined_file_path = os.path.join(output_dir, "all_valid_solutions.png")
    plt.savefig(combined_file_path, dpi=300)
    plt.show()


# Function to create and save a high-resolution network graph using spring layout
def create_solution_network_graph(valid_groups, pattern_df):
    G = nx.Graph()
    solution_artifacts = []

    # Process each solution and add edges based on shared artifacts
    for idx, (start, end) in enumerate(valid_groups):
        artifacts = pattern_df["Artifacts"].values[start:end]
        artifact_list = set(artifact.strip() for group in artifacts for artifact in group.split(','))
        solution_artifacts.append((f"Solution {idx + 1}", artifact_list))
        G.add_node(f"Solution {idx + 1}")

    # Compare each solution to find shared artifacts and add edges with labels
    for i, (solution1, artifacts1) in enumerate(solution_artifacts):
        for j in range(i + 1, len(solution_artifacts)):
            solution2, artifacts2 = solution_artifacts[j]
            shared_artifacts = artifacts1.intersection(artifacts2)
            if shared_artifacts:
                G.add_edge(solution1, solution2, weight=len(shared_artifacts), label=', '.join(shared_artifacts))

    # Use spring layout for clear separation
    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, seed=42, k=2)  # Increase k to add spacing between clusters
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=150)  # Smaller nodes
    nx.draw_networkx_edges(G, pos, width=[w * 0.3 for w in weights])  # Thinner edges
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    # Draw edge labels with artifact names in common
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Save the network graph with high resolution
    network_file_path = os.path.join(output_dir, "solution_network_graph.png")
    plt.title("Solution Co-Occurrence Network")
    plt.savefig(network_file_path, dpi=300, bbox_inches="tight")  # High-resolution output
    plt.show()


# Apply DP to find largest valid groups and save solutions
largest_valid_groups = dp_largest_valid_groups(pattern_groups)
solution_files = save_individual_solutions_v2(largest_valid_groups, pattern_groups)
plot_all_solutions_v2(solution_files)

# Generate and save the high-resolution solution network graph
create_solution_network_graph(largest_valid_groups, pattern_groups)