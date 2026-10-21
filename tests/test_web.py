import pytest
from fastapi.testclient import TestClient

from src.causal_extraction.spans import SentenceTooLong, decode_spans
from src.web.app import create_app
from src.web.service import analyze_text, split_sentences


def span(text, confidence=0.8):
    return {'text': text, 'confidence': confidence, 'start': 0, 'end': len(text)}


class FakePredictor:
    def predict_detailed(self, text):
        if text.startswith('Long'):
            raise SentenceTooLong('Too long')
        if text.startswith('Inflation'):
            return {'causes': [span('Inflation', 0.9)], 'effects': [span('poverty', 0.7)]}
        if text.startswith('Multiple'):
            return {'causes': [span('A'), span('B')], 'effects': [span('C'), span('D')]}
        if text.startswith('Partial'):
            return {'causes': [span('Partial')], 'effects': []}
        return {'causes': [], 'effects': []}


def test_sentence_splitting_abbreviations_decimals_and_paragraphs():
    text = 'Dr. Smith raised rates by 2.5 percent. Markets fell!\n\nA separate paragraph'
    assert split_sentences(text) == [
        'Dr. Smith raised rates by 2.5 percent.', 'Markets fell!', 'A separate paragraph']


def test_statistics_skip_and_incomplete_extraction():
    result = analyze_text('Inflation increases poverty. No change. Partial cause. Long sentence.', FakePredictor())
    assert result['statistics'] == {'total_sentences': 4, 'analyzed_sentences': 3,
                                    'causal_sentences': 1, 'non_causal_sentences': 2,
                                    'skipped_sentences': 1}
    assert result['rows'][0]['confidence'] == pytest.approx(0.8)
    assert result['rows'][0]['sentence_number'] == 1
    assert result['skipped'][0]['sentence_number'] == 4


def test_multiple_fragments_do_not_invent_pairs():
    result = analyze_text('Multiple effects.', FakePredictor())
    assert len(result['rows']) == 1
    assert result['rows'][0]['ambiguous'] is True
    assert len(result['rows'][0]['causes']) == 2
    assert len(result['rows'][0]['effects']) == 2


def test_original_text_and_confidence_decoding():
    result = decode_spans('High taxes hurt growth.', [(0, 4, 'B-CAUSE', 0.9),
                          (5, 10, 'E-CAUSE', 0.7), (11, 15, 'O', 0.9),
                          (16, 22, 'S-EFFECT', 0.6)])
    assert [item['text'] for item in result] == ['High taxes', 'growth']
    assert result[0]['confidence'] == pytest.approx(0.8)


def test_malformed_bioes_never_joins_different_types():
    result = decode_spans('A B', [(0, 1, 'I-CAUSE', 0.8), (2, 3, 'E-EFFECT', 0.7)])
    assert [(item['type'], item['text']) for item in result] == [('CAUSE', 'A'), ('EFFECT', 'B')]


def test_api_page_validation_and_single_model_load():
    calls = []

    def factory():
        calls.append(True)
        return FakePredictor()

    with TestClient(create_app(factory)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        for text in ['', '   ', 'Русский текст.', 'x' * 200001]:
            assert client.post('/api/analyze', json={'text': text}).status_code == 422
        assert calls == []
        for _ in range(2):
            response = client.post('/api/analyze', json={'text': 'Inflation increases poverty.'})
            assert response.status_code == 200
            assert response.json()['statistics']['causal_sentences'] == 1
        assert len(calls) == 1
        assert client.post('/api/analyze', json={'text': 'Test. ' * 2001}).status_code == 422


def test_model_load_failure_is_retryable():
    calls = []

    def factory():
        calls.append(True)
        if len(calls) == 1:
            raise FileNotFoundError('missing weights')
        return FakePredictor()

    with TestClient(create_app(factory)) as client:
        assert client.post('/api/analyze', json={'text': 'Test.'}).status_code == 503
        assert client.post('/api/analyze', json={'text': 'Test.'}).status_code == 200


def test_analyzer_only_page_and_routes():
    import re
    with TestClient(create_app(FakePredictor)) as client:
        page = client.get('/').text
        script = client.get('/static/app.js').text
        assert 'analysis-form' in page and 'type="file"' in page
        for marker in ('class="tabs"', 'generation-panel', 'training-panel', 'GigaChat', 'Colab'):
            assert marker not in page
        assert '/api/generator/' not in script and '/api/colab/' not in script
        ids = set(re.findall(r'id="([^"]+)"', page))
        assert set(re.findall(r"\$\('([^']+)'\)", script)) <= ids
        for url in ('/api/generator/status', '/api/generator/health', '/api/colab/bundle', '/api/colab/notebook'):
            assert client.get(url).status_code == 404
        for url in ('/api/generator/start', '/api/generator/cancel'):
            assert client.post(url, json={}).status_code == 404


def test_text_and_file_analysis_match():
    with TestClient(create_app(FakePredictor)) as client:
        text = 'Inflation increases poverty. No change.'
        expected = client.post('/api/analyze', json={'text': text}).json()
        uploaded = client.post('/api/analyze-file', files={'file': ('sample.txt', text.encode(), 'text/plain')})
        assert uploaded.status_code == 200
        result = uploaded.json()
        assert result.pop('source') == 'sample.txt'
        assert result.pop('warnings') == []
        assert result == expected


def test_subtokens_are_combined_and_long_input_is_not_silently_truncated():
    import torch
    from types import SimpleNamespace
    from src.causal_extraction.predict import CausalPredictor

    class Encoding(dict):
        def word_ids(self, index):
            return [None, 0, 0, 1, None]

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs['truncation'] is False
            return Encoding(input_ids=torch.tensor([[101, 1, 2, 3, 102]]),
                            attention_mask=torch.ones((1, 5), dtype=torch.long),
                            offset_mapping=torch.tensor([[(0, 0), (0, 3), (3, 9), (10, 14), (0, 0)]]))

    class Model:
        def __call__(self, input_ids, attention_mask):
            logits = torch.zeros((1, 5, 9))
            for token, label in enumerate([0, 4, 4, 8, 0]):
                logits[0, token, label] = 5
            return SimpleNamespace(logits=logits)

    predictor = CausalPredictor.__new__(CausalPredictor)
    predictor.tokenizer, predictor.model = Tokenizer(), Model()
    predictor.device = torch.device('cpu')
    predictor.config = {'training': {'max_length': 128}}
    result = predictor.predict_detailed('Unrestful fall')
    assert [item['text'] for item in result['causes']] == ['Unrestful']
    assert [item['text'] for item in result['effects']] == ['fall']
    assert 0 < result['causes'][0]['confidence'] < 1
    predictor.config['training']['max_length'] = 4
    with pytest.raises(SentenceTooLong):
        predictor.predict_detailed('Unrestful fall')
