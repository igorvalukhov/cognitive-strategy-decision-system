import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .metrics import LABELS

LABEL2ID = {label: index for index, label in enumerate(LABELS)}


class CausalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        path = Path(data_path)
        content = path.read_text(encoding='utf-8-sig')
        if path.suffix == '.jsonl':
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            records = []
            for block in content.strip().split('\n\n'):
                entries = [line.rsplit('\t', 1) for line in block.splitlines() if line.strip()]
                if entries:
                    records.append({'tokens': [e[0] for e in entries], 'labels': [e[1] for e in entries]})
        self.items, self.records = [], records
        for number, record in enumerate(records, 1):
            tokens, tags = record['tokens'], record['labels']
            if not tokens or len(tokens) != len(tags) or any(tag not in LABEL2ID for tag in tags):
                raise ValueError(f'Invalid annotation at {path}:{number}')
            encoding = tokenizer(tokens, is_split_into_words=True, truncation=False)
            if len(encoding['input_ids']) > max_length:
                raise ValueError(f'Example {number} in {path} exceeds {max_length} tokens; not silently truncated')
            seen, labels = set(), []
            for word_id in encoding.word_ids():
                if word_id is None or word_id in seen:
                    labels.append(-100)
                else:
                    labels.append(LABEL2ID[tags[word_id]])
                    seen.add(word_id)
            pad = max_length-len(labels)
            self.items.append({'input_ids': torch.tensor(encoding['input_ids']+[tokenizer.pad_token_id]*pad),
                               'attention_mask': torch.tensor(encoding['attention_mask']+[0]*pad),
                               'labels': torch.tensor(labels+[-100]*pad)})
        if not self.items:
            raise ValueError(f'Empty dataset: {path}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def split_paths(directory):
    directory = Path(directory)
    paths = {}
    for split in ('train', 'val', 'test'):
        candidates = [directory / (split + ext) for ext in ('.jsonl', '.txt')]
        existing = [p for p in candidates if p.is_file()]
        if not existing:
            raise ValueError(f'Missing {split}.jsonl or {split}.txt in {directory}')
        paths[split] = existing[0]
    return paths
