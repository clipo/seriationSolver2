import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set
import os


def parse_seriation_results(results_file: str) -> Dict[str, Set[str]]:
    """
    Parse the seriation results file and extract assemblage names for each sequence.

    Parameters:
    - results_file: Path to seriation_results.txt

    Returns:
    - Dictionary mapping sequence names to sets of assemblage names
    """
    groups = {}
    current_group = None
    reading_assemblages = False

    print("Parsing seriation results file...")

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect sequence header
            if line.startswith('Sequence_') and line.endswith(':'):
                current_group = line[:-1]  # Remove trailing colon
                groups[current_group] = set()
                reading_assemblages = False
                print(f"\nFound {current_group}")

            # Start collecting assemblages
            elif line == "Assemblages:":
                reading_assemblages = True

            # Ignore lines with "Length" and collect assemblage names
            elif reading_assemblages and line and "Length" not in line:
                assemblage = line.strip()
                if assemblage:  # Ensure it is not an empty line
                    groups[current_group].add(assemblage)
                    print(f"  Added assemblage: {assemblage}")

    # Print summary of what we found
    print("\nSummary of parsed sequences:")
    for group, assemblages in groups.items():
        print(f"\n{group}:")
        print(f"  Assemblages: {', '.join(sorted(assemblages))}")

    return groups


def create_seriation_network(results_file: str, output_dir: str = 'seriation_results'):
    """
    Create a network visualization showing relationships between seriation groups with minimal edge crossings.
    """
    # Parse the results file
    groups = parse_seriation_results(results_file)

    # Create network
    G = nx.Graph()

    # Add nodes for groups
    for group, assemblages in groups.items():
        G.add_node(group, node_type='group', assemblages=assemblages)

    # Find and add edges for shared assemblages
    print("\nFinding shared assemblages between sequences:")
    for group1 in groups:
        for group2 in groups:
            if group1 < group2:  # Avoid duplicate edges
                shared = groups[group1].intersection(groups[group2])
                if shared:  # Only create edge if there are shared assemblages
                    print(f"\n{group1} and {group2} share assemblages:")
                    print(f"  {', '.join(sorted(shared))}")
                    G.add_edge(group1, group2,
                               weight=len(shared),
                               shared_assemblages=shared)

    # Create visualization
    plt.figure(figsize=(15, 15))

    # Calculate node sizes and edge widths
    node_sizes = [len(G.nodes[node]['assemblages']) * 1000 for node in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]  # Edge width based on shared assemblages

    # Apply Kamada-Kawai layout for reduced edge crossings
    pos = nx.kamada_kawai_layout(G)

    # Draw network
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color='lightblue',
                           alpha=0.7)

    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,  # Dynamic edge width based on shared assemblages
                           alpha=0.5,
                           edge_color='gray')

    # Add labels
    node_labels = {node: f"{node}\n({len(G.nodes[node]['assemblages'])} assemblages)"
                   for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    edge_labels = {(u, v): '\n'.join(sorted(G[u][v]['shared_assemblages']))
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edge_labels,
                                 font_size=8)

    plt.title("Seriation Groups Network (Kamada-Kawai Layout)\nNode size = number of assemblages\nEdge width = number of shared assemblages",
              pad=20)
    plt.axis('off')

    # Save visualization
    network_path = os.path.join(output_dir, 'seriation_network.png')
    plt.savefig(network_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nNetwork visualization saved to: {network_path}")

    # Print network statistics
    print("\nNetwork Statistics:")
    print(f"Number of sequences: {len(G.nodes())}")
    print(f"Number of connections: {len(G.edges())}")

    return G

def add_network_visualization(self):
    """
    Add method to DPSeriationSolver class to create network visualization
    """

    def create_network(self, output_dir: str = 'seriation_results'):
        """Create network visualization of seriation groups"""
        results_file = os.path.join(output_dir, 'seriation_results.txt')
        if os.path.exists(results_file):
            return create_seriation_network(results_file, output_dir)
        else:
            print("Results file not found. Run fit() first.")
            return None

    # Add method to class
    DPSeriationSolver.create_network = create_network


if __name__ == "__main__":
    results_file = "seriation_results/seriation_results.txt"
    G = create_seriation_network(results_file)

    # Print all connections found
    print("\nDetailed connections:")
    for edge in G.edges(data=True):
        print(f"\n{edge[0]} - {edge[1]}:")
        print(f"  {len(edge[2]['shared_assemblages'])} shared assemblages:")
        print(f"  {', '.join(sorted(edge[2]['shared_assemblages']))}")