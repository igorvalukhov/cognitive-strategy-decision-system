import torch.nn as nn
from transformers import AutoConfig, AutoModelForTokenClassification

class CausalExtractor(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super(CausalExtractor, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.config.hidden_dropout_prob = dropout
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            config=self.config
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        return outputs 