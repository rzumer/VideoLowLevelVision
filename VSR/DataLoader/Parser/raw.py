#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

import copy
import os
from pathlib import Path

import numpy as np

from . import _logger, parse_index
from ..VirtualFile import RawFile
from ...Util.ImageProcess import imresize, imcompress, shrink_to_multiple_scale


def _lr_file_from_hr(vf):
    #vf_lr = copy.copy(vf)
    new_path = ''
    head, tail = os.path.split(vf.path)

    if vf.path.is_file():
        fname = tail
        new_path = os.path.join(head, 'lr')
        new_path = Path(os.path.join(new_path, fname))
        #vf_lr = RawFile(new_path, 'YV12', (1920, 1080))
        vf_lr = RawFile(new_path, 'YV12', vf.size)
        #vf_lr.path = new_path
        #vf_lr.file = [new_path]
        #vf_lr.length[new_path.name] = new_path.stat().st_size
    else:
        exit(1)
        '''
        new_path = os.path.join(head, 'lr')
        vf_lr.path = Path(new_path)
        for _file in vf_lr.file:
            new_path = _file
            head, tail = os.path.split(_file)
            if tail != '':
                fname = head
                head, tail = os.path.split(head)
                new_path = os.path.join(head, 'lr')
                new_path = Path(os.path.join(new_path, fname))
            vf_lr.length[_file.name] = new_path.stat().st_size
        '''
    #bp()
    return vf_lr


class Parser(object):
  def __init__(self, dataset, config):
    urls = dataset.get(config.method, [])
    self.file_objects = [
      RawFile(fp, dataset.mode, (dataset.width, dataset.height)) for fp in urls]
    self.scale = config.scale
    self.depth = config.depth
    self.method = config.method
    self.modcrop = config.modcrop
    self.resample = config.resample
    if config.convert_to.lower() in ('gray', 'l'):
      self.color_format = 'L'
    elif config.convert_to.lower() in ('yuv', 'ycbcr'):
      self.color_format = 'YCbCr'
    elif config.convert_to.lower() in ('rgb',):
      self.color_format = 'RGB'
    else:
      _logger.warning('Use grayscale by default. '
                      'Unknown format {}'.format(config.convert_to))
      self.color_format = 'L'
    if self.depth < 0:
      self.depth = 2 ** 31 - 1
    # calculate index range
    n_frames = []
    for _f in self.file_objects:
      l = _f.frames
      if l < self.depth:
        n_frames.append(1)
      else:
        n_frames.append(l - self.depth + 1)
    index = np.arange(int(np.sum(n_frames)))
    self.index = [parse_index(i, n_frames) for i in index]

  def __getitem__(self, index):
    vf_lr = None

    if isinstance(index, slice):
      ret = []
      for key, seq in self.index[index]:
        vf = self.file_objects[key]
        if self.scale == 1:
            vf_lr = _lr_file_from_hr(vf)
        ret += self.gen_frames(copy.deepcopy(vf), copy.deepcopy(vf_lr), seq)
      return ret
    else:
      key, seq = self.index[index]
      vf = self.file_objects[key]
      if self.scale == 1:
          vf_lr = _lr_file_from_hr(vf)
      return self.gen_frames(copy.deepcopy(vf), copy.deepcopy(vf_lr), seq)

  def __len__(self):
    return len(self.index)

  def gen_frames(self, vf, vf_lr, index):
    assert isinstance(vf, RawFile)

    _logger.debug(f'Prefetching {vf.name} @{index}')
    vf.reopen()
    depth = self.depth
    depth = min(depth, vf.frames)
    vf.seek(index)
    if vf_lr is not None:
        vf_lr.reopen()
        vf_lr.seek(index)
    hr = [shrink_to_multiple_scale(img, self.scale)
          if self.modcrop else img for img in vf.read_frame(depth)]
    if vf_lr is not None:
        lr = [shrink_to_multiple_scale(img, self.scale)
          if self.modcrop else img for img in vf_lr.read_frame(depth)]
    elif all(scale == 1 for scale in self.scale):
        lr = [imcompress(img, random.randint(10, 60)) for img in frames_hr]
    else:
        lr = [imresize(img,
                   np.reciprocal(self.scale, dtype='float32'),
                   resample=self.resample)
          for img in hr]
    hr = [img.convert(self.color_format) for img in hr]
    lr = [img.convert(self.color_format) for img in lr]
    return [(hr, lr, (vf.name, index, vf.frames))]

  @property
  def capacity(self):
    # bytes per pixel
    bpp = 1.5 * (1 + np.reciprocal(self.scale, dtype='float32'))
    # NOTE use uint64 to prevent sum overflow
    return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64)
                   for vf in self.file_objects])
