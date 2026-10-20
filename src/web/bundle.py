import io
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def make_colab_bundle(root, dataset_dir=None):
    root = Path(root)
    dataset = Path(dataset_dir) if dataset_dir else root/'data'
    paths = []
    for split in ('train', 'val', 'test'):
        candidates = [dataset/(split+ext) for ext in ('.jsonl', '.txt')]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError(f'Нет {split}.jsonl или {split}.txt в папке датасета.')
        paths.append(path)
    output = io.BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name in ('__init__.py','train.py','train_module.py','training_data.py','metrics.py','model.py','utils.py'):
            archive.write(root/'src/causal_extraction'/name, 'src/causal_extraction/'+name)
        archive.write(root/'configs/training.yaml', 'configs/training.yaml')
        for path in paths:
            archive.write(path, 'dataset/'+path.name)
        archive.writestr('requirements-colab.txt', 'torch>=2.6\ntransformers>=4.35,<5\nscikit-learn>=1.3\npyyaml>=6\nnumpy>=1.24\n')
    return output.getvalue()
