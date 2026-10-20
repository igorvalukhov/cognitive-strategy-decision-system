import torch
import yaml
import json
from pathlib import Path
from transformers import AutoTokenizer
from .model import CausalExtractor
from .utils import extract_spans_from_bioes
from .spans import LABELS, SentenceTooLong, decode_spans

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class CausalPredictor:
    """The class for inference (prediction) of cause-and-effect relationships"""
    
    def __init__(self, model_path='models/best_model.pt', config_path='configs/default.yaml'):
        active = PROJECT_ROOT / 'models/active_model.json'
        if model_path == 'models/best_model.pt' and active.exists():
            selection = json.loads(active.read_text(encoding='utf-8'))
            model_path, config_path = selection['weights'], selection['config']
        model_path = PROJECT_ROOT / model_path
        config_path = PROJECT_ROOT / config_path
        if not model_path.is_file():
            raise FileNotFoundError(f'Model weights not found: {model_path}')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'], use_fast=True)
        
        self.model = CausalExtractor(
            self.config['model']['name'], 
            self.config['model']['num_labels'],
            dropout=self.config['model'].get('dropout', 0.3)
        )
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"The model was uploaded from the device: {self.device}")

    def predict_detailed(self, text):
        """Full-sentence extraction with source offsets and uncalibrated scores.

        Subtokens are aggregated to words because training assigns the same
        BIOES label to every subtoken of an original word.
        """
        first_subtoken = self.config.get('inference', {}).get('word_strategy') == 'first_subtoken'
        if first_subtoken:
            from src.synthetic.dataset import words_with_offsets
            source_words = words_with_offsets(text)
            encoding = self.tokenizer([w[0] for w in source_words], is_split_into_words=True,
                                      return_offsets_mapping=True, return_tensors='pt', truncation=False)
        else:
            encoding = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=False)
        limit = self.config['training']['max_length']
        if encoding['input_ids'].shape[1] > limit:
            raise SentenceTooLong(f'Предложение длиннее {limit} токенов модели. Разбейте его на более короткие.')
        offsets = encoding.pop('offset_mapping')[0].tolist()
        word_ids = encoding.word_ids(0)
        with torch.inference_mode():
            output = self.model(encoding['input_ids'].to(self.device),
                                attention_mask=encoding['attention_mask'].to(self.device))
            probabilities = output.logits[0].softmax(dim=-1).cpu()
        groups = []
        for index, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if not groups or groups[-1][0] != word_id:
                groups.append((word_id, []))
            groups[-1][1].append(index)
        words = []
        for word_id, indices in groups:
            mean_probs = probabilities[indices[0]] if first_subtoken else probabilities[indices].mean(dim=0)
            label_id = int(mean_probs.argmax())
            start, end = (source_words[word_id][1:]) if first_subtoken else (offsets[indices[0]][0], offsets[indices[-1]][1])
            words.append((start, end,
                          LABELS[label_id], float(mean_probs[label_id])))
        spans = decode_spans(text, words)
        return {'text': text, 'causes': [s for s in spans if s['type'] == 'CAUSE'],
                'effects': [s for s in spans if s['type'] == 'EFFECT']}
        
    def predict(self, text):
        if self.config.get('inference', {}).get('word_strategy') == 'first_subtoken':
            result = self.predict_detailed(text)
            return {'text': text, 'causes': [s['text'] for s in result['causes']],
                    'effects': [s['text'] for s in result['effects']]}
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
