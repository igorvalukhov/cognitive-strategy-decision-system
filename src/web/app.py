import logging
from typing import Literal
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import analyze_text
from .documents import MAX_FILE_BYTES, extract_document
from .jobs import GenerationJobs, ROOT
from .bundle import make_colab_bundle
from src.synthetic.generator import LocalGigaChat

STATIC_DIR = Path(__file__).parent / 'static'
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200000)


class GenerationRequest(BaseModel):
    provider: Literal['gigachat', 'qwen'] = 'gigachat'
    domain: str = Field(min_length=2, max_length=200)
    count: int = Field(default=2000, ge=30, le=10000)
    negative_ratio: float = Field(default=.3, ge=.1, le=.5)
    seed: int = Field(default=42, ge=0, le=1000000)


def create_app(predictor_factory=None):
    app = FastAPI(title='Анализ причин и следствий')
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    app.state.predictor = None
    inference_lock = Lock()
    jobs = GenerationJobs()

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / 'index.html')

    @app.get('/api/generator/status')
    def generation_status():
        return jobs.status()

    @app.get('/api/generator/health')
    def generation_health(provider: Literal['gigachat', 'qwen'] = 'gigachat'):
        try:
            return LocalGigaChat(provider=provider).health()
        except Exception as error:
            return {'available': False, 'message': str(error)}

    @app.post('/api/generator/start', status_code=202)
    def generation_start(payload: GenerationRequest):
        if not payload.domain.strip():
            raise HTTPException(422, 'Укажите домен.')
        try:
            return jobs.start(payload.model_dump())
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.post('/api/generator/cancel')
    def generation_cancel():
        try:
            return jobs.cancel()
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get('/api/colab/bundle')
    def colab_bundle(dataset: str = "data"):
        selected = (ROOT / dataset).resolve()
        if not selected.is_relative_to((ROOT / "data").resolve()):
            raise HTTPException(422, "Выберите папку внутри data/.")
        try:
            return Response(make_colab_bundle(ROOT, selected), media_type='application/zip',
                            headers={'Content-Disposition': 'attachment; filename="module1-training.zip"'})
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get('/api/colab/notebook')
    def colab_notebook():
        return FileResponse(ROOT/'notebooks/Module1_Training.ipynb', filename='Module1_Training.ipynb')

    @app.post('/api/analyze-file')
    def analyze_file(file: UploadFile = File(...)):
        try:
            contents = file.file.read(MAX_FILE_BYTES+1)
            document = extract_document(file.filename or 'upload', contents)
            result = analyze(AnalyzeRequest(text=document['text']))
            result.update(source=document['filename'], warnings=document['warnings'])
            return result
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            file.file.close()

    @app.post('/api/analyze')
    def analyze(payload: AnalyzeRequest):
        text = payload.text.strip()
        if not text:
            raise HTTPException(422, 'Введите текст для анализа.')
        # Current weights are English-only; do not present unsupported Russian
        # input as a successful analysis with zero causal relationships.
        if any('\u0400' <= char <= '\u04ff' for char in text):
            raise HTTPException(422, 'Эта модель обучена на английском языке. Вставьте английский текст.')
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(409, 'Модель занята другим анализом. Попробуйте ещё раз чуть позже.')
        try:
            if app.state.predictor is None:
                try:
                    if predictor_factory is None:
                        from src.causal_extraction.predict import CausalPredictor
                        app.state.predictor = CausalPredictor()
                    else:
                        app.state.predictor = predictor_factory()
                except Exception as error:
                    logger.exception('Failed to load causal model')
                    raise HTTPException(503, 'Не удалось загрузить модель. Проверьте models/best_model.pt, '
                                        'зависимости и доступ к Hugging Face при первом запуске. '
                                        'Подробности — в журнале сервера.') from error
            return analyze_text(text, app.state.predictor)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except HTTPException:
            raise
        except Exception as error:
            logger.exception('Analysis failed')
            raise HTTPException(500, 'Ошибка анализа. Подробности — в журнале сервера.') from error
        finally:
            inference_lock.release()

    return app


app = create_app()
