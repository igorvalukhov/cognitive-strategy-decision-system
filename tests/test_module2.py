import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cognitive_map import CognitiveMapBuilder, CognitiveMapAnalyzer

def test_module2():
    print("Testing Cognitive Map Module (Module 2)...")
    
    mock_predictions = [
        {'causes': ['The sudden increase in interest rates'], 'effects': ['a massive outflow of foreign investment']},
        {'causes': ['Interest rates increased'], 'effects': ['foreign investment outflow']}, # То же самое, но другими словами (проверка нормализации)
        {'causes': ['Political instability'], 'effects': ['stock market crash', 'inflation']},
        {'causes': ['Inflation'], 'effects': ['reduced purchasing power']}
    ]
    
    builder = CognitiveMapBuilder()
    builder.add_relations(mock_predictions)
    graph = builder.get_graph()
    
    print(f"\nGraph stats: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    analyzer = CognitiveMapAnalyzer(graph)
    
    print("\nRoot Causes (in_degree == 0):")
    for cause in analyzer.get_root_causes():
        print(f"  - {cause}")
        
    print("\nTarget Effects (out_degree == 0):")
    for effect in analyzer.get_target_effects():
        print(f"  - {effect}")
        
    print("\nStrongest Links (by frequency):")
    for u, v, weight in analyzer.get_strongest_links(k=3):
        print(f"  {u}  --->  {v}  (weight: {weight})")

    print("\nModule 2 tests completed successfully!")

if __name__ == '__main__':
    test_module2()