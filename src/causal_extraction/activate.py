"""Activate a checked training output directory without replacing baseline weights."""
import argparse
import json
from pathlib import Path
from .predict import CausalPredictor, PROJECT_ROOT
from src.synthetic.dataset import atomic_json


def activate(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((PROJECT_ROOT/'models').resolve()):
        raise ValueError('Place the training output under the project models directory first.')
    metrics = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    weights, config = directory/'best_model.pt', directory/'config.yaml'
    predictor = CausalPredictor(str(weights), str(config))
    predictor.predict_detailed('Higher taxes caused public protests in the capital.')
    atomic_json(PROJECT_ROOT/'models/active_model.json', {
        'weights': weights.relative_to(PROJECT_ROOT).as_posix(),
        'config': config.relative_to(PROJECT_ROOT).as_posix(),
        'test_metrics': metrics['test']})
    print('Activated. Restart the web server to load this model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    activate(parser.parse_args().directory)
