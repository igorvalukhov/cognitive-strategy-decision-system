import networkx as nx

class CognitiveMapAnalyzer:
    """Analyzes the topological properties of a cognitive map."""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        
    def get_root_causes(self) -> list[str]:
        """Returns nodes with in_degree == 0 (sources/root causes)."""
        return [node for node, degree in self.graph.in_degree() if degree == 0]
        
    def get_target_effects(self) -> list[str]:
        """Returns nodes with out_degree == 0 (final consequences)."""
        return [node for node, degree in self.graph.out_degree() if degree == 0]
        
    def get_top_centrality_nodes(self, k: int = 5) -> list[tuple[str, float]]:
        """Returns the top K nodes by intermediary centrality (betweenness)."""
        if len(self.graph) == 0:
            return []
        centrality = nx.betweenness_centrality(self.graph, weight='weight', normalized=True)
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:k]
        
    def get_strongest_links(self, k: int = 5) -> list[tuple[str, str, int]]:
        """Returns the top K most frequent cause-and-effect relationships."""
        edges = self.graph.edges(data=True)
        sorted_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
        return [(u, v, data['weight']) for u, v, data in sorted_edges[:k]]
        
    def export_to_graphml(self, filepath: str):
        """Exports the graph to GraphML format for visualization in Gephi."""
        nx.write_graphml(self.graph, filepath)
        print(f"Graph exported to {filepath}")