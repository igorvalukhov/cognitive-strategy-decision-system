import json
import os
import random
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .dataset import atomic_json, is_near_duplicate, parse_annotation, save_splits

MODEL_REPO = 'ai-sage/GigaChat3.1-10B-A1.8B-GGUF'
MODEL_ALIAS = 'gigachat-local'
DEFAULT_URL = 'http://127.0.0.1:8081/v1'
POSITIVE_STYLES = ['cause before effect', 'effect before cause', 'causality without because or therefore',
                   'negative consequence', 'conditional or possible consequence', 'policy intervention',
                   'economic mechanism', 'institutional decision']
NEGATIVE_STYLES = ['temporal sequence without stated causation', 'correlation only',
                   'a quoted organization name containing a causal-looking word',
                   'two unrelated facts', 'explicit denial of a causal connection']


class LocalGigaChat:
    def __init__(self, base_url=None, model=None, provider='gigachat'):
        if provider not in ('gigachat', 'qwen'):
            raise ValueError('Unknown generator provider')
        self.provider = provider
        prefix = provider.upper()
        fallback_url = DEFAULT_URL if provider == 'gigachat' else 'http://127.0.0.1:11434/v1'
        fallback_model = MODEL_ALIAS if provider == 'gigachat' else 'qwen2.5:7b'
        self.base_url = (base_url or os.getenv(prefix + '_BASE_URL', fallback_url)).rstrip('/')
        self.model = model or os.getenv(prefix + '_MODEL', fallback_model)
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username:
            raise ValueError('Invalid GIGACHAT_BASE_URL')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, endpoint, payload=None, timeout=240):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + endpoint, data=data, headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(req, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            raise ConnectionError(f'Сервер генерации недоступен по {self.base_url}: {error}') from error

    def health(self):
        result = self._request('/models', timeout=10)
        if self.model not in [r.get('id') for r in result.get('data', [])]:
            raise ConnectionError(f'Сервер доступен, но модель {self.model} не найдена. Проверьте имя модели в настройках провайдера.')
        return {'available': True, 'base_url': self.base_url, 'configured_model': self.model,
                'models': [r.get('id') for r in result.get('data', [])]}

    def complete(self, prompt, seed):
        response = self._request('/chat/completions', {
            'model': self.model, 'messages': [
                {'role': 'system', 'content': 'You generate accurately annotated English training data. Return only the requested JSON object. No reasoning or markdown.'},
                {'role': 'user', 'content': prompt}],
            'temperature': .65, 'top_p': .9, 'max_tokens': 600, 'seed': seed,
            'response_format': {'type': 'json_object'}})
        choice = response['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise ValueError('Model response was truncated')
        content = choice['message'].get('content') or ''
        content = content.strip()
        if content.startswith('```'):
            content = content.split('\n', 1)[-1].rsplit('```', 1)[0]
        return json.loads(content)


def generate(domain, count, negative_ratio, seed, directory, client, tokenizer, notify, cancel):
    directory.mkdir(parents=True, exist_ok=True)
    records, rejected, attempts = [], {}, 0
    rng = random.Random(seed)
    kinds = [False] * round(count * negative_ratio)
    kinds += [True] * (count-len(kinds))
    rng.shuffle(kinds)
    client.health()
    notify(accepted=0, attempts=0, rejected={}, message='Модель подключена. Генерация и проверка примеров.')
    while len(records) < count and attempts < count * 6:
        if cancel.is_set():
            return {'cancelled': True, 'accepted': len(records)}
        causal = kinds[len(records)]
        style = rng.choice(POSITIVE_STYLES if causal else NEGATIVE_STYLES)
        attempt_seed = seed + attempts
        attempts += 1
        task = ('Include exactly one cause and one effect. Wrap COMPLETE causal events in <CAUSE> and <EFFECT> tags. '
                'Include verbs, negation, direction of change, relevant quantities and modality inside spans. '
                'No nested tags. A mere noun phrase is not a complete event when a change is stated.' if causal else
                'The sentence must NOT assert any cause-effect relationship, including implicit causation. Do not use tags.')
        prompt = (f'Domain (topic only): {json.dumps(domain)}. Write ONE self-contained English sentence of 12-45 words. '
                  f'Style: {style}. {task} Use diverse institutions, events and mechanisms. '
                  f'Variation seed: {attempt_seed}. Return {{"marked_text":"..."}}.')
        raw = None
        try:
            raw = client.complete(prompt, attempt_seed)
            record = parse_annotation(raw['marked_text'], causal)
            if len(tokenizer(record['tokens'], is_split_into_words=True)['input_ids']) > 128:
                raise ValueError('Too many model tokens')
            if is_near_duplicate(record, records):
                raise ValueError('Duplicate or near duplicate')
            review = client.complete(
                'Check this training example independently. Does the sentence state a cause-effect relation? '
                'For a positive example, do the marked spans capture the full cause and full effect including '
                'negation and direction of change, with correct roles? For a negative example there must be NO '
                'asserted causal relation. Return {"valid":true/false,"reason":"short explanation"}. '
                f'Expected causal={str(causal).lower()}. Example: {json.dumps(raw["marked_text"])}', attempt_seed + 1000000)
            if review.get('valid') is not True:
                raise ValueError('Rejected by model review')
            record.update(domain=domain, style=style, generation_seed=attempt_seed,
                          model=client.model, provider=getattr(client, 'provider', 'unspecified'), server_model=client.model, marked_text=raw['marked_text'],
                          review=review, annotation_source='synthetic_model_review_not_human_gold')
            records.append(record)
            with (directory / 'accepted.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        except (ValueError, KeyError, TypeError, IndexError) as error:
            reason = str(error)[:160]
            rejected[reason] = rejected.get(reason, 0) + 1
            with (directory / 'rejected.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'attempt': attempts, 'reason': reason, 'response': raw}, ensure_ascii=False)+'\n')
        notify(accepted=len(records), attempts=attempts, rejected=rejected,
               message=f'Принято {len(records)} из {count}; попыток {attempts}.')
    if len(records) != count:
        raise ValueError(f'Получено только {len(records)}/{count}. Частичные данные сохранены, набор не активирован.')
    splits = save_splits(records, directory, seed)
    manifest = {'version': 1, 'domain': domain, 'language': 'en', 'count': count,
                'negative_ratio': negative_ratio, 'seed': seed, 'splits': splits,
                'generator': client.model, 'provider': getattr(client, 'provider', 'unspecified'), 'server_model': client.model, 'base_url': client.base_url,
                'attempts': attempts, 'rejected': rejected, 'human_reviewed': False,
                'note': 'Validation/test are synthetic, not an independent real-world benchmark.'}
    atomic_json(directory / 'manifest.json', manifest)
    return manifest
