import logging
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import analyze_text
from .documents import MAX_FILE_BYTES, extract_document

STATIC_DIR = Path(__file__).parent / 'static'
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200000)


def create_app(predictor_factory=None):
    app = FastAPI(title='Анализ причин и следствий')
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    app.state.predictor = None
    inference_lock = Lock()

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / 'index.html')

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
