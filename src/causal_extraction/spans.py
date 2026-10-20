"""Decode word-level BIOES predictions while preserving the original text."""

LABELS = ('O', 'B-CAUSE', 'I-CAUSE', 'E-CAUSE', 'S-CAUSE',
          'B-EFFECT', 'I-EFFECT', 'E-EFFECT', 'S-EFFECT')


def decode_spans(text, words):
    """words: (start, end, BIOES label, mean subtoken label probability).

    Malformed I/E sequences are recovered as new spans; entity types never mix.
    Confidence is an uncalibrated mean over words, not a relation probability.
    """
    spans, active = [], None

    def finish():
        nonlocal active
        if active is not None:
            scores = active.pop('_scores')
            active['text'] = text[active['start']:active['end']]
            active['confidence'] = sum(scores) / len(scores)
            spans.append(active)
            active = None

    for start, end, label, score in words:
        if label == 'O':
            finish()
            continue
        prefix, kind = label.split('-', 1)
        if prefix in ('B', 'S') or active is None or active['type'] != kind:
            finish()
            active = {'type': kind, 'start': start, 'end': end, '_scores': []}
        active['end'] = end
        active['_scores'].append(float(score))
        if prefix in ('E', 'S'):
            finish()
    finish()
    return spans


class SentenceTooLong(ValueError):
    pass
