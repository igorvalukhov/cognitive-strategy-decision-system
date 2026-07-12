import torch
import yaml
from transformers import AutoTokenizer
from src.model import CausalExtractor
from src.utils import extract_spans_from_bioes


class CausalPredictor:
    """The class for inference (prediction) of cause-and-effect relationships"""
    
    def __init__(self, model_path='models/best_model.pt', config_path='configs/default.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        self.model = CausalExtractor(
            self.config['model']['name'], 
            self.config['model']['num_labels'],
            dropout=self.config['model'].get('dropout', 0.3)
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"The model was uploaded from the device: {self.device}")
        
    def predict(self, text):
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.config['training']['max_length'], 
            padding='max_length', 
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy().tolist()
            
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        spans = extract_spans_from_bioes(tokens, preds)
        
        causes = [s['text'] for s in spans if s['type'] == 'CAUSE']
        effects = [s['text'] for s in spans if s['type'] == 'EFFECT']
        
        return {
            'text': text,
            'causes': causes,
            'effects': effects
        }


if __name__ == '__main__':
    predictor = CausalPredictor()
    
    test_sentences = [
        "The sudden increase in interest rates led to a massive outflow of foreign investment.",
        "Political instability in the region caused a stock market crash.",
        "The Senate reviewed the fiscal reform report but deferred action until next quarter."
    ]
    
    print("\n" + "="*60)
    print(" TESTING THE MODEL")
    print("="*60)
    
    for text in test_sentences:
        result = predictor.predict(text)
        print(f"\n text: {result['text']}")
        print(f"   causes: {result['causes']}")
        print(f"   effects: {result['effects']}")
    
    print("\n" + "="*60)