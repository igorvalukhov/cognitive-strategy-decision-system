import hashlib
import json
import random
import re
from pathlib import Path

TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def words_with_offsets(text):
    return [(match.group(), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def parse_annotation(marked_text, causal):
    """Compute offsets in CLEAN text while removing markup, never reuse raw offsets."""
    if not isinstance(marked_text, str) or not 15 <= len(marked_text) <= 1800:
        raise ValueError('Invalid sentence length')
    raw = marked_text.strip()
    if re.search(r'[\u0400-\u04ff]', raw):
        raise ValueError('The target model requires English examples')
    tags = re.compile(r'</?(CAUSE|EFFECT)>')
    pieces, spans, opened, cursor, length = [], [], None, 0, 0
    for match in tags.finditer(raw):
        part = raw[cursor:match.start()]
        pieces.append(part)
        length += len(part)
        kind = match.group(1)
        if match.group().startswith('</'):
            if opened is None or opened[0] != kind:
                raise ValueError('Unbalanced tags')
            spans.append({'type': kind, 'start': opened[1], 'end': length})
            opened = None
        else:
            if opened is not None:
                raise ValueError('Nested tags are not supported')
            opened = (kind, length)
        cursor = match.end()
    if opened is not None:
        raise ValueError('Unclosed tag')
    pieces.append(raw[cursor:])
    text = ''.join(pieces)
    if '<' in text or '>' in text or '\n' in text:
        raise ValueError('Extra markup or multiple lines')
    if causal:
        if sorted(s['type'] for s in spans) != ['CAUSE', 'EFFECT']:
            raise ValueError('Exactly one cause and one effect are required')
    elif spans:
        raise ValueError('Non-causal examples must have no tags')
    words = words_with_offsets(text)
    if not 5 <= len(words) <= 75:
        raise ValueError('Use between 5 and 75 words/punctuation tokens')
    labels = ['O'] * len(words)
    for span in spans:
        indices = [i for i, (_, start, end) in enumerate(words)
                   if start >= span['start'] and end <= span['end']]
        if not indices or words[indices[0]][1] != span['start'] or words[indices[-1]][2] != span['end']:
            raise ValueError('Span must align with word boundaries and have no edge whitespace')
        for j, index in enumerate(indices):
            prefix = 'S' if len(indices) == 1 else 'B' if j == 0 else 'E' if j == len(indices)-1 else 'I'
            labels[index] = f"{prefix}-{span['type']}"
        span['text'] = text[span['start']:span['end']]
    normalized = ' '.join(token.lower() for token, _, _ in words if token.isalnum())
    return {'id': hashlib.sha256(normalized.encode()).hexdigest()[:20], 'text': text,
            'tokens': [word for word, _, _ in words], 'labels': labels, 'spans': spans,
            'causal': causal}


def is_near_duplicate(record, accepted):
    tokens = set(re.findall(r'\w+', record['text'].lower()))
    for other in accepted:
        if record['id'] == other['id']:
            return True
        previous = set(re.findall(r'\w+', other['text'].lower()))
        if len(tokens & previous) / max(1, len(tokens | previous)) >= .85:
            return True
    return False


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_splits(records, directory, seed=42):
    """Stratified 80/10/10 split after global duplicate rejection."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for causal in (False, True):
        group = [r for r in records if r['causal'] == causal]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * .1))
        n_test = max(1, round(len(group) * .1))
        if len(group) < n_val + n_test + 1:
            raise ValueError('Too few examples of each class for independent splits')
        splits['val'].extend(group[:n_val])
        splits['test'].extend(group[n_val:n_val+n_test])
        splits['train'].extend(group[n_val+n_test:])
    for name, values in splits.items():
        rng.shuffle(values)
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values), encoding='utf-8')
        (directory / f'{name}.txt').write_text('\n\n'.join('\n'.join(f'{t}\t{l}' for t,l in zip(r['tokens'], r['labels'])) for r in values)+'\n', encoding='utf-8')
    return {name: len(values) for name, values in splits.items()}
