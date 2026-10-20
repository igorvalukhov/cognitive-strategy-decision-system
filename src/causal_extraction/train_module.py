"""Validation selects checkpoints; a held-out test is evaluated only at the end."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from .training_data import CausalDataset
from .metrics import evaluate_sequences
from .model import CausalExtractor
from .utils import set_seed

ROOT = Path(__file__).resolve().parents[2]


def evaluate_model(model, loader, device):
    predictions, references = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            predictions.extend(output.logits.argmax(-1).cpu().tolist())
            references.extend(batch['labels'].tolist())
    return evaluate_sequences(predictions, references)


def train(config_path=None, dataset_dir=None, output_dir=None):
    config_path = Path(config_path) if config_path else ROOT/'configs/training.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if dataset_dir is None:
        dataset_dir = ROOT/'data'
    dataset_dir = Path(dataset_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output_dir = Path(output_dir) if output_dir else ROOT/'models/runs'/stamp
    params = config['training']
    set_seed(params['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], use_fast=True)
    from .training_data import split_paths
    paths = split_paths(dataset_dir)
    datasets = {split: CausalDataset(paths[split], tokenizer, params['max_length'])
                for split in ('train', 'val', 'test')}
    identities = {name: {tuple(t.lower() for t in r['tokens']) for r in data.records} for name,data in datasets.items()}
    if identities['train'] & identities['val'] or identities['train'] & identities['test'] or identities['val'] & identities['test']:
        raise ValueError('Duplicate texts across dataset splits')
    output_dir.mkdir(parents=True, exist_ok=False)
    loaders = {name: DataLoader(data, batch_size=params['batch_size'], shuffle=name=='train', num_workers=0)
               for name,data in datasets.items()}
    model = CausalExtractor(config['model']['name'], 9, dropout=config['model']['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params['learning_rate']), weight_decay=params['weight_decay'])
    steps = len(loaders['train'])*params['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*params['warmup_ratio']), steps)
    best, stale, history = -1., 0, []
    config['inference'] = {'word_strategy': 'first_subtoken'}
    (output_dir/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    manifest = {'dataset': str(dataset_dir), 'device': str(device), 'config': config,
                'dataset_sha256': {s: hashlib.sha256(paths[s].read_bytes()).hexdigest() for s in datasets},
                'torch_version': torch.__version__}
    (output_dir/'run.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0.
        for batch in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            total_loss += float(output.loss.detach())
        metrics = evaluate_model(model, loaders['val'], device)
        record = {'epoch': epoch+1, 'loss': total_loss/len(loaders['train']), 'validation': metrics}
        history.append(record)
        (output_dir/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        score = metrics['exact_span']['f1']
        print(f'Epoch {epoch+1}: loss={record["loss"]:.4f}, exact-span F1={score:.4f}, token F1={metrics["token_macro"]["f1"]:.4f}', flush=True)
        if score > best:
            best, stale = score, 0
            torch.save(model.state_dict(), output_dir/'best_model.pt')
        else:
            stale += 1
            if stale >= params['patience']:
                break
    model.load_state_dict(torch.load(output_dir/'best_model.pt', map_location=device, weights_only=True))
    test_metrics = evaluate_model(model, loaders['test'], device)
    (output_dir/'metrics.json').write_text(json.dumps({'best_validation_span_f1': best, 'test': test_metrics,
        'caution': 'Metrics describe the supplied held-out test only; data origin is not assumed.'}, indent=2), encoding='utf-8')
    print('TEST (best validation checkpoint):', json.dumps(test_metrics, indent=2), flush=True)
    print('Saved:', output_dir, flush=True)
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--dataset')
    parser.add_argument('--output')
    args = parser.parse_args()
    train(args.config, args.dataset, args.output)
