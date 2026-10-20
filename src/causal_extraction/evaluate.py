"""Evaluate a trained module-1 run using its own tokenization configuration."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from .model import CausalExtractor
from .training_data import CausalDataset
from .train_module import evaluate_model


def evaluate(run, dataset):
    run = Path(run)
    config = yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    if config.get('inference', {}).get('word_strategy') != 'first_subtoken':
        raise ValueError('This evaluator expects a new module-1 training run.')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], use_fast=True)
    data = CausalDataset(Path(dataset)/'test.jsonl', tokenizer, config['training']['max_length'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalExtractor(config['model']['name'], 9).to(device)
    model.load_state_dict(torch.load(run/'best_model.pt', map_location=device, weights_only=True))
    metrics = evaluate_model(model, DataLoader(data, batch_size=16), device)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    evaluate(args.run, args.dataset)
