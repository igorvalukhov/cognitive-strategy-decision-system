import io
from zipfile import ZipFile
from pathlib import Path
from src.web.bundle import make_colab_bundle
from src.synthetic.generator import LocalGigaChat


def test_bundle_without_generator_or_manifest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for split in ('train', 'val', 'test'):
        (tmp_path/(split+'.txt')).write_text('Word\tO\n', encoding='utf-8')
    with ZipFile(io.BytesIO(make_colab_bundle(root, tmp_path))) as archive:
        names = archive.namelist()
        assert 'dataset/train.txt' in names
        assert not any('synthetic' in n or 'manifest' in n for n in names)
        code = archive.read('src/causal_extraction/train_module.py').decode()
        assert 'data/generated' not in code
        assert 'manifest.json' not in code


def test_provider_selection(monkeypatch):
    monkeypatch.setenv('GIGACHAT_MODEL', 'gigachat-annotator')
    monkeypatch.setenv('QWEN_MODEL', 'qwen-test')
    monkeypatch.setenv('QWEN_BASE_URL', 'http://localhost:11434/v1')
    assert LocalGigaChat(provider='gigachat').model == 'gigachat-annotator'
    client = LocalGigaChat(provider='qwen')
    assert client.model == 'qwen-test'
    assert client.base_url == 'http://localhost:11434/v1'
