import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from .model import CausalExtractor
from .data_preparation import CausalDataset

def train(config_path='configs/default.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Starting training on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    train_dataset = CausalDataset(config['data']['train_path'], tokenizer, config['training']['max_length'])
    val_dataset = CausalDataset(config['data']['val_path'], tokenizer, config['training']['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    model = CausalExtractor(
        config['model']['name'], 
        config['model']['num_labels'], 
        dropout=config['model'].get('dropout', 0.3)
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = int(total_steps * config['training'].get('warmup_ratio', 0.1))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())
                
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        
        mask = (labels_arr != 0) & (labels_arr != -100)
        p_flat = preds_arr[mask]
        l_flat = labels_arr[mask]
        
        valid_labels = list(range(1, 9))
        if np.isin(l_flat, valid_labels).any():
            _, _, f1, _ = precision_recall_fscore_support(
                l_flat, p_flat, 
                labels=valid_labels, 
                average='macro', 
                zero_division=0
            )
        else:
            f1 = 0.0
            
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {avg_loss:.4f} | Val Macro F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['output']['model_dir'], 'best_model.pt'))
            print(f" Saved new best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print("Early stopping triggered!")
                break
    
    print(f"\nTraining completed! Best Val Macro F1: {best_f1:.4f}")

if __name__ == '__main__':
    train()