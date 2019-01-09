"""
Unit test for DataLoader.VirtualFile
"""
import os

if not os.getcwd().endswith('UTest'):
    os.chdir('UTest')
from VLLV.DataLoader.VirtualFile import *
from VLLV.DataLoader.Dataset import *
from VLLV.Util.ImageProcess import img_to_array

DATASETS = load_datasets('./data/fake_datasets.yml')
RAW = 'data/raw.yv12'
IMG = 'data/set5_x2/img_001_SRF_2_LR.png'


def test_raw_seek():
    vf = RawFile(RAW, 'YV12', [32, 32])
    f1 = vf.read_frame(1)[0]
    vf.seek(0, SEEK_SET)
    f2 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_CUR)
    f3 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_END)
    f4 = vf.read_frame(1)[0]
    vf.seek(-2, SEEK_END)
    vf.seek(1, SEEK_CUR)
    f5 = vf.read_frame(1)[0]

    F = [f1, f2, f3, f4, f5]
    F = [img_to_array(f) for f in F]
    assert np.all(F[0] == F[1])
    assert np.all(F[1] == F[2])
    assert np.all(F[3] == F[4])


def test_image_seek():
    vf = ImageFile(IMG, False)
    f1 = vf.read_frame(1)[0]
    vf.seek(0, SEEK_SET)
    f2 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_CUR)
    f3 = vf.read_frame(1)[0]
    vf.seek(-1, SEEK_END)
    f4 = vf.read_frame(1)[0]
    vf.seek(-2, SEEK_END)
    f5 = vf.read_frame(1)[0]

    F = [f1, f2, f3, f4, f5]
    F = [img_to_array(f) for f in F]
    assert np.all(F[0] == F[1])
    assert np.all(F[1] == F[2])
    assert np.all(F[3] == F[4])


def test_vf_copy():
    import copy
    vf0 = ImageFile(IMG, False)
    vf1 = copy.deepcopy(vf0)
    vf0.read_frame(1)
    try:
        vf0.read_frame(1)
    except EOFError:
        pass
    vf1.read_frame(1)
