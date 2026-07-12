import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import CausalPredictor

def run_tests():
    print("Running Causal Module Tests...")
    
    if not os.path.exists('models/best_model.pt'):
        print("Error: models/best_model.pt not found. Train the model first!")
        return
        
    predictor = CausalPredictor()
    
    test_cases = [
        "Inflation reduces the purchasing power of the population.",
        "The introduction of sanctions hit export revenues hard.",
        "A simple sentence without any causal links."
    ]
    
    for text in test_cases:
        res = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"   Cause: {res['causes']}")
        print(f"   Effect: {res['effects']}")
        
    print("\nTests completed successfully!")

if __name__ == '__main__':
    run_tests()