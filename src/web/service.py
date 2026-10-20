import re

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from src.causal_extraction.spans import SentenceTooLong

MAX_SENTENCES = 2000


def split_sentences(text):
    # No NLTK downloads needed. Paragraph boundaries also terminate sentences.
    parameters = PunktParameters()
    parameters.abbrev_types = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
                               'e.g', 'i.e', 'vs', 'etc', 'u.s', 'u.k', 'no', 'fig'}
    tokenizer = PunktSentenceTokenizer(parameters)
    return [sentence.strip() for paragraph in re.split(r'\n\s*\n', text.strip())
            for sentence in tokenizer.tokenize(paragraph) if sentence.strip()]


def analyze_text(text, predictor):
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError('Введите текст для анализа.')
    if len(sentences) > MAX_SENTENCES:
        raise ValueError(f'За один раз можно обработать не более {MAX_SENTENCES} предложений.')
    rows, skipped = [], []
    for number, sentence in enumerate(sentences, 1):
        try:
            prediction = predictor.predict_detailed(sentence)
        except SentenceTooLong as error:
            skipped.append({'sentence_number': number, 'sentence': sentence, 'reason': str(error)})
            continue
        causes, effects = prediction['causes'], prediction['effects']
        if causes and effects:
            spans = causes + effects
            rows.append({'sentence_number': number, 'sentence': sentence,
                         'causes': causes, 'effects': effects,
                         'confidence': sum(s['confidence'] for s in spans) / len(spans),
                         'ambiguous': len(causes) > 1 or len(effects) > 1})
    analyzed = len(sentences) - len(skipped)
    return {'statistics': {'total_sentences': len(sentences), 'analyzed_sentences': analyzed,
                           'causal_sentences': len(rows),
                           'non_causal_sentences': analyzed - len(rows),
                           'skipped_sentences': len(skipped)},
            'rows': rows, 'skipped': skipped}
