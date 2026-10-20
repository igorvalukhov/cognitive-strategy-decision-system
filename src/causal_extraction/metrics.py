"""False positives on O are included; exact spans supplement tag accuracy."""
from sklearn.metrics import precision_recall_fscore_support

LABELS = ['O', 'B-CAUSE', 'I-CAUSE', 'E-CAUSE', 'S-CAUSE', 'B-EFFECT', 'I-EFFECT', 'E-EFFECT', 'S-EFFECT']


def spans_from_ids(ids):
    spans, current = set(), None
    for index, label_id in enumerate(list(ids) + [0]):
        label = LABELS[int(label_id)]
        if label == 'O':
            if current:
                spans.add((current[0], current[1], index))
                current = None
            continue
        prefix, kind = label.split('-')
        if prefix in ('B', 'S') or current is None or current[0] != kind:
            if current:
                spans.add((current[0], current[1], index))
            current = (kind, index)
        if prefix in ('E', 'S'):
            spans.add((current[0], current[1], index+1))
            current = None
    return spans


def prf(tp, fp, fn):
    precision = tp / (tp+fp) if tp+fp else 0.
    recall = tp / (tp+fn) if tp+fn else 0.
    return {'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.}


def evaluate_sequences(predictions, references):
    truth, guesses = [], []
    tp = fp = fn = pair_tp = pair_fp = pair_fn = 0
    sentence_tp = sentence_fp = sentence_fn = 0
    negative_count = negative_false_alarms = 0
    for pred, ref in zip(predictions, references):
        selected = [(int(p), int(r)) for p,r in zip(pred,ref) if r != -100]
        p, r = [v[0] for v in selected], [v[1] for v in selected]
        truth.extend(r)
        guesses.extend(p)
        actual, found = spans_from_ids(r), spans_from_ids(p)
        tp += len(actual & found)
        fp += len(found - actual)
        fn += len(actual - found)
        actual_pair = {s[0] for s in actual} == {'CAUSE', 'EFFECT'}
        found_pair = {s[0] for s in found} == {'CAUSE', 'EFFECT'}
        sentence_tp += int(actual_pair and found_pair)
        sentence_fp += int(not actual_pair and found_pair)
        sentence_fn += int(actual_pair and not found_pair)
        if actual_pair and len(actual) == 2:
            exact = actual == found
            pair_tp += int(exact)
            pair_fn += int(not exact)
            pair_fp += int(bool(found) and not exact)
        elif not actual:
            pair_fp += int(bool(found))
        if not actual:
            negative_count += 1
            negative_false_alarms += int(bool(found))
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, guesses, labels=list(range(1,9)), average='macro', zero_division=0)
    return {'token_macro': {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)},
            'exact_span': prf(tp, fp, fn), 'exact_single_pair': prf(pair_tp, pair_fp, pair_fn),
            'causal_sentence': prf(sentence_tp, sentence_fp, sentence_fn),
            'negative_examples': negative_count, 'negative_false_alarms': negative_false_alarms,
            'negative_false_alarm_rate': negative_false_alarms/negative_count if negative_count else None,
            'evaluated_sentences': len(references)}
