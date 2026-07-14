import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data_preparation import CausalDataset
from src.model import CausalExtractor
from src.utils import compute_metrics

def evaluate(config_path='configs/default.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running the final assessment on a test set (device: {device})")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    test_dataset = CausalDataset(config['data']['test_path'], tokenizer, config['training']['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    model_path = os.path.join(config['output']['model_dir'], 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"Error: The model was not found on the way {model_path}. First, start the train.py!")
        return
        
    model = CausalExtractor(config['model']['name'], config['model']['num_labels']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs['logits'], dim=-1)
            
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            
    metrics = compute_metrics(all_preds, all_labels)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS (TEST SET)")
    print("="*50)
    print(f" Macro F1:   {metrics['f1']:.4f}")
    print(f" Precision:  {metrics['precision']:.4f}")
    print(f" Recall:     {metrics['recall']:.4f}")
    print("="*50)

if __name__ == '__main__':
    evaluate()