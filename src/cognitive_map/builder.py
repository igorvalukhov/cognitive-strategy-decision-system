import networkx as nx
from .normalizer import ConceptNormalizer

class CognitiveMapBuilder:
    """Builds an oriented weighted graph of cause-effect relationships."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.normalizer = ConceptNormalizer()
        
    def add_relations(self, predictions: list[dict]):
        """
        Adds connections to the graph.
        predictions: the list of dictionaries looks like {'causes': ['...'], 'effects': ['...']}
        """
        for pred in predictions:
            causes = pred.get('causes', [])
            effects = pred.get('effects', [])
            
            for cause in causes:
                for effect in effects:
                    norm_cause = self.normalizer.normalize(cause)
                    norm_effect = self.normalizer.normalize(effect)
                    
                    if not norm_cause or not norm_effect:
                        continue
                        
                    if self.graph.has_edge(norm_cause, norm_effect):
                        self.graph[norm_cause][norm_effect]['weight'] += 1
                    else:
                        self.graph.add_edge(norm_cause, norm_effect, weight=1)
                        
    def get_graph(self) -> nx.DiGraph:
        return self.graph
        
    def clear(self):
        self.graph.clear()