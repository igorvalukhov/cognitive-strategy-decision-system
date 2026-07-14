import torch
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support

LABELS = ['O', 'B-CAUSE', 'I-CAUSE', 'E-CAUSE', 'S-CAUSE', 
          'B-EFFECT', 'I-EFFECT', 'E-EFFECT', 'S-EFFECT']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(pred_labels, true_labels):
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    mask = (true_labels != 0) & (true_labels != -100)
    if mask.sum() == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels[mask], 
        pred_labels[mask], 
        average='macro', 
        zero_division=0
    )
    return {'f1': float(f1), 'precision': float(precision), 'recall': float(recall)}

def extract_spans_from_bioes(tokens, labels):
    spans = []
    current_span = None
    
    for i, label_id in enumerate(labels):
        if label_id == -100:
            continue
        label = ID2LABEL.get(label_id, 'O')
        token = tokens[i]
        
        if token in ['[PAD]', '[CLS]', '[SEP]', '[UNK]']:
            continue
            
        if label.startswith('B-') or label.startswith('S-'):
            if current_span:
                spans.append(current_span)
            span_type = label.split('-')[1]
            text_part = token[2:] if token.startswith('##') else token
            current_span = {'type': span_type, 'text': text_part}
            
            if label.startswith('S-'):
                spans.append(current_span)
                current_span = None
                
        elif label.startswith('I-') or label.startswith('E-'):
            if current_span:
                if token.startswith('##'):
                    current_span['text'] += token[2:]
                else:
                    current_span['text'] += ' ' + token
                    
                if label.startswith('E-'):
                    spans.append(current_span)
                    current_span = None
        else:
            if current_span:
                spans.append(current_span)
                current_span = None
                
    if current_span:
        spans.append(current_span)
        
    return spans