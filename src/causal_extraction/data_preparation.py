import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils import LABEL2ID

class CausalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = [l.strip() for l in block.split('\n') if '\t' in l]
            if not lines:
                continue
                
            tokens = [l.split('\t')[0] for l in lines]
            tags = [LABEL2ID.get(l.split('\t')[1], 0) for l in lines]
            
            self.examples.append({
                'tokens': tokens,
                'tags': tags
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        tokens = item['tokens']
        tags = item['tags']
        
        encoding = self.tokenizer(
            tokens, 
            is_split_into_words=True, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        word_ids = encoding.word_ids(0)
        labels = [-100 if w is None else tags[w] for w in word_ids]
        
        encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['labels'].squeeze(0)
        }