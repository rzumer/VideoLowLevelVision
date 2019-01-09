import os

if not os.getcwd().endswith('UTest'):
    os.chdir('UTest')
from VLLV.DataLoader.Dataset import _glob_absolute_pattern, load_datasets

DATASETS = load_datasets('./data/fake_datasets.yml')


def test_glob_absolute_pattern():
    URL = './data/set5_x2'
    node = _glob_absolute_pattern(URL)
    assert len(node) == 5
    assert node[0].match('img_001_SRF_2_LR.png')
    assert node[1].match('img_002_SRF_2_LR.png')
    assert node[2].match('img_003_SRF_2_LR.png')
    assert node[3].match('img_004_SRF_2_LR.png')
    assert node[4].match('img_005_SRF_2_LR.png')

    URL = './data/flying_chair/**/*.flo'
    node = _glob_absolute_pattern(URL)
    assert len(node) == 1
    assert node[0].match('0000.flo')

    URL = './data/**/*.png'
    node = _glob_absolute_pattern(URL)
    assert len(node) == 10


def test_existence():
    _K = DATASETS.keys()
    for k in _K:
        print('==== [', k, '] ====')
        _V = []
        try:
            _V = DATASETS[k].train
        except ValueError:
            if not _V:
                print('[Warning] Train set of', k, 'doesn\'t exist.')
        finally:
            _V = []
        try:
            _V = DATASETS[k].val
        except ValueError:
            if not _V:
                print('[Warning] Val set of', k, 'doesn\'t exist.')
        finally:
            _V = []
        try:
            _V = DATASETS[k].test
        except ValueError:
            if not _V:
                print('[Warning] Test set of', k, 'doesn\'t exist.')
        print('=========================', flush=True)
