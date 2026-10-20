import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.synthetic.dataset import atomic_json
from src.synthetic.generator import LocalGigaChat, generate

ROOT = Path(__file__).resolve().parents[2]


class GenerationJobs:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.current = None

    def status(self):
        with self.lock:
            if self.current:
                return dict(self.current)
        path = self.root/'data/generated/job.json'
        if path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            if result['status'] in ('running', 'cancelling'):
                result.update(status='interrupted', message='Сервер перезапущен. Частичные примеры сохранены; начните новую генерацию.')
            return result
        return {'status': 'idle'}

    def update(self, **values):
        with self.lock:
            self.current.update(values)
            atomic_json(self.root/'data/generated/job.json', self.current)

    def start(self, parameters):
        with self.lock:
            if self.current and self.current['status'] in ('running', 'cancelling'):
                raise ValueError('Генерация уже идёт.')
            run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            self.cancel_event.clear()
            self.current = {'id': run_id, 'status': 'running', **parameters, 'accepted': 0,
                            'attempts': 0, 'message': 'Подключение к выбранной модели…'}
            atomic_json(self.root/'data/generated/job.json', self.current)
        threading.Thread(target=self._run, args=(run_id, parameters), daemon=True).start()
        return self.status()

    def _run(self, run_id, parameters):
        directory = self.root/'data/generated'/run_id
        try:
            parameters = dict(parameters)
            provider = parameters.pop('provider', 'gigachat')
            client = LocalGigaChat(provider=provider)
            client.health()
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', use_fast=True)
            result = generate(**parameters, directory=directory, client=client, tokenizer=tokenizer,
                              notify=self.update, cancel=self.cancel_event)
            if result.get('cancelled'):
                self.update(status='cancelled', message='Генерация отменена; частичные данные сохранены.')
            else:
                atomic_json(self.root/'data/generated/latest.json', {'id': run_id, 'path': directory.relative_to(self.root).as_posix()})
                self.update(status='completed', manifest=result, message='Набор готов. Укажите его папку во вкладке обучения.')
        except Exception as error:
            self.update(status='failed', message=str(error))

    def cancel(self):
        with self.lock:
            if not self.current or self.current['status'] not in ('running', 'cancelling'):
                raise ValueError('Нет активной генерации.')
            self.cancel_event.set()
            self.current.update(status='cancelling', message='Остановка после текущего ответа модели…')
        return self.status()
