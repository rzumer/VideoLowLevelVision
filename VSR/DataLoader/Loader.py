"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang, Raphaël Zumer
Email: wenyi.tang@intel.com, rzumer@tebako.net
Created Date: May 8th 2018
Updated Date: April 2nd 2019

Load frames with specified filter in given directories,
and provide inheritable API for specific loaders.

changelog 2019-4-2
- Introduce `parser` to deal with more & more complex data distribution

changelog 2018-8-29
- Added BasicLoader and QuickLoader (multiprocessor loader)
- Deprecated BatchLoader (and Loader)
"""

#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/4 下午2:42

import importlib
import threading as th

import numpy as np
from psutil import virtual_memory

from . import _logger
from ..Util import Utility
from ..Util.Config import Config
from ..Util.ImageProcess import crop


def _augment(image, op):
  """Image augmentation"""
  if op[0]:
    image = np.rot90(image, 1)
  if op[1]:
    image = np.fliplr(image)
  if op[2]:
    image = np.flipud(image)
  return image


class EpochIterator:
  """An iterator for generating batch data in one epoch

  Args:
      loader: A `BasicLoader` or `QuickLoader` to provide properties.
      grids: A list of tuple, commonly returned from
        `BasicLoader._generate_crop_grid`.
  """

  def __init__(self, loader, grids):
    self.batch = loader.batch
    self.scale = loader.scale
    self.aug = loader.aug
    self.loader = loader
    self.grids = grids

  def __len__(self):
    t = len(self.grids)
    b = self.batch
    return t // b + int(np.ceil((t % b) / b))

  def __iter__(self):
    return self

  def __next__(self):
    batch_hr, batch_lr, batch_name = [], [], []
    if not self.grids:
      raise StopIteration

    while self.grids and len(batch_hr) < self.batch:
      hr, lr, box, name = self.grids.pop(0)
      box = np.array(box, 'int32')
      box_lr = box // [*self.scale, *self.scale]
      # if self.loader.method == 'train':
      #   assert (np.mod(box, [*self.scale, *self.scale]) == 0).all()
      crop_hr = [crop(img, box) for img in hr]
      crop_lr = [crop(img, box_lr) for img in lr]
      ops = np.random.randint(0, 2, [3]) if self.aug else [0, 0, 0]
      clip_hr = [_augment(img, ops) for img in crop_hr]
      clip_lr = [_augment(img, ops) for img in crop_lr]
      batch_hr.append(np.stack(clip_hr))
      batch_lr.append(np.stack(clip_lr))
      batch_name.append(name)

    if batch_hr and batch_lr and batch_name:
      try:
        batch_hr = np.squeeze(np.stack(batch_hr), 1)
        batch_lr = np.squeeze(np.stack(batch_lr), 1)
      except ValueError:
        # squeeze error
        batch_hr = np.stack(batch_hr)
        batch_lr = np.stack(batch_lr)
      batch_name = np.stack(batch_name)

    if np.ndim(batch_hr) == 3:
      batch_hr = np.expand_dims(batch_hr, -1)
    if np.ndim(batch_lr) == 3:
      batch_lr = np.expand_dims(batch_lr, -1)

    return batch_hr, batch_lr, batch_name, batch_lr


class BasicLoader:
  """Basic loader in single thread

  Args:
      dataset: A `Dataset` to load by this loader.
      method: A string in ('train', 'val', 'test') specifies which subset to
        use in the dataset. Also 'train' set will shuffle buffers each epoch.
      config: A `Config` class, including 'batch', 'depth', 'patch_size',
        'scale', 'steps_per_epoch' and 'convert_to' arguments.
      augmentation: A boolean to specify whether call `_augment` to batches.
        `_augment` will randomly flip or rotate images.
      kwargs: override config key-values.
  """

  def __init__(self, dataset, method, config, augmentation=False, **kwargs):
    config = self._parse_config(config, **kwargs)
    config.method = method.lower()
    parser = dataset.get('parser', 'default_parser')
    _logger.debug(f"Parser: [{parser}]")
    try:
      _m = importlib.import_module(parser)
    except ImportError:
      _m = importlib.import_module(f'.{parser}', 'VSR.DataLoader.Parser')
    self.parser = _m.Parser(dataset, config)
    self.aug = augmentation
    if hasattr(self.parser, 'color_format'):
      self.color_format = self.parser.color_format
    else:
      self.color_format = 'RGB'
    # self.pair = getattr(dataset, '{}_pair'.format(method))
    self.loaded = 0
    self.frames = []  # a list of tuple represents (HR, LR, name) of a clip

  def _parse_config(self, config: Config, **kwargs):
    _config = Config(config)
    _config.update(kwargs)
    _needed_args = ('batch', 'depth', 'scale',
                    'steps_per_epoch', 'convert_to', 'modcrop')
    for _arg in _needed_args:
      # Set default and check values
      if _arg not in _config:
        if _arg in ('batch', 'scale'):
          raise ValueError(_arg + ' is required in config.')
        elif _arg == 'depth':
          _config.depth = 1
        elif _arg == 'steps_per_epoch':
          _config.steps_per_epoch = -1
        elif _arg == 'convert_to':
          _config.convert_to = 'RGB'
        elif _arg == 'modcrop':
          _config.modcrop = True
    self.depth = _config.depth
    self.patch_size = _config.patch_size
    self.scale = Utility.to_list(_config.scale, 2)
    self.patches_per_epoch = _config.steps_per_epoch * _config.batch
    self.batch = _config.batch
    self.crop = _config.crop
    self.modcrop = _config.modcrop
    self.resample = _config.resample
    return _config

  def _generate_crop_grid(self, frames, size, shuffle=False):
    """generate randomly cropped box of `frames`

    Args:
        frames: a list of tuple, commonly returned from `_process_at_file`.
        size: an int scalar to specify number of generated crops.
        shuffle: a boolean, whether to shuffle the outputs.

    Return:
        list of tuple: containing (HR, LR, box, name) respectively,
          where HR and LR are reference frames, box is a list of 4
          int of crop coordinates.
    """
    if not frames:
      _logger.warning('frames is empty. [size={}]'.format(size))
      return []
    patch_size = Utility.to_list(self.patch_size, 2)
    patch_size = Utility.shrink_mod_scale(patch_size, self.scale)
    if size < 0:
      index = np.arange(len(frames)).tolist()
      size = len(frames)
    else:
      if self.crop == 'random':
        index = np.random.randint(len(frames), size=size).tolist()
      else:
        index = np.arange(size).tolist()
    grids = []
    for i, (hr, lr, name) in enumerate(frames):
      _w, _h = hr[0].width, hr[0].height
      if self.crop in ('not', 'none') or self.crop is None:
        _pw, _ph = _w, _h
      else:
        _pw, _ph = patch_size
      amount = index.count(i)
      if self.crop == 'random':
        x = np.random.randint(0, _w - _pw + 1, size=amount)
        y = np.random.randint(0, _h - _ph + 1, size=amount)
      elif self.crop == 'center':
        x = np.array([(_w - _pw) // 2] * amount)
        y = np.array([(_h - _ph) // 2] * amount)
      elif self.crop == 'stride':
        _x = np.arange(0, _w - _pw + 1, _pw)
        _y = np.arange(0, _h - _ph + 1, _ph)
        x, y = np.meshgrid(_x, _y)
        x = x.flatten()
        y = y.flatten()
      else:
        x = np.zeros([amount])
        y = np.zeros([amount])
      x -= x % self.scale[0]
      y -= y % self.scale[1]
      grids += [(hr, lr, [_x, _y, _x + _pw, _y + _ph], name)
                for _x, _y in zip(x, y)]
    if shuffle:
      np.random.shuffle(grids)
    return grids[:size]

  def _prefetch(self, memory_usage=None, shard=1, index=0):
    """Prefetch `size` files and load into memory. Specify `shard` will
    divide loading files into `shard` shards in order to execute in
    parallel.

    NOTE: parallelism is implemented via `QuickLoader`

    Args:
      memory_usage: desired virtual memory to use, could be int (bytes) or
        a readable string ('3GB', '1TB'). Default to use all available
        memories.
      shard: an int scalar to specify the number of shards operating in
        parallel.
      index: an int scalar, representing shard index
    """

    if self.loaded & (1 << index):
      return
    # check memory usage
    if isinstance(memory_usage, str):
      memory_usage = Utility.str_to_bytes(memory_usage)
    free_memory = virtual_memory().available
    if not memory_usage:
      memory_usage = free_memory
    memory_usage = np.min(
      [np.uint64(memory_usage), free_memory])
    if hasattr(self.parser, 'capacity'):
      cap = self.parser.capacity
    else:
      cap = -1
    if cap <= memory_usage:
      # load all clips
      interval = int(np.ceil(len(self.parser) / shard))
      if index == shard - 1:
        frames = self.parser[index * interval:]
      else:
        frames = self.parser[index * interval:(index + 1) * interval]
      self.frames += frames
      self.loaded |= (1 << index)
    else:
      scale_factor = 0.9
      prop = memory_usage / cap / shard * scale_factor
      # How many frames can be read into memory each thread each epoch
      # Note: we assume each "frame" has a close size.
      n = max(1, int(np.round(len(self.parser) * prop)))  # at least 1 sample
      frames = []
      for i in np.random.permutation(len(self.parser))[:n]:
        frames += self.parser[i]
      self.frames += frames

  def make_one_shot_iterator(self, memory_usage=None, shuffle=False):
    """make an `EpochIterator` to enumerate batches of the dataset

    Args:
        memory_usage: desired virtual memory to use, could be int (bytes) or
          a readable string ('3GB', '1TB'). Default to use all available
          memories.
        shuffle: A boolean whether to shuffle the patch grids.

    Returns:
        An EpochIterator
    """
    self._prefetch(memory_usage, 1, 0)
    grids = self._generate_crop_grid(self.frames, self.patches_per_epoch,
                                     shuffle=shuffle)
    if not (self.loaded == 1):
      self.frames.clear()
    return EpochIterator(self, grids)

    def __init__(self, dataset, method, config, augmentation=False, **kwargs):
        config = self._parse_config(config, **kwargs)
        self.file_names = dataset.__getattr__(method.lower()) or []
        self.method = method
        self.flow = dataset.flow
        self.aug = augmentation
        if config.convert_to.lower() in ('gray', 'l'):
            self.color_format = 'L'
        elif config.convert_to.lower() in ('yuv', 'ycbcr'):
            self.color_format = 'YCbCr'
        elif config.convert_to.lower() in ('rgb',):
            self.color_format = 'RGB'
        else:
            tf.logging.warning(
                f'Unknown format {config.convert_to}, use grayscale by default')
            self.color_format = 'L'
        self.loaded = 0
        self.free_memory_on_start = virtual_memory().free
        self.frames = []  # a list of tuple represents (HR, LR, name) of a clip
        self.prob = self._read_file(dataset)._calc_select_prob()

    def _parse_config(self, config: Config, **kwargs):
        _config = Config(config)
        _config.update(kwargs)
        _needed_args = ('batch', 'depth', 'scale',
                        'steps_per_epoch', 'convert_to', 'modcrop')
        for _arg in _needed_args:
            # Set default and check values
            if _arg not in _config:
                if _arg in ('batch', 'scale'):
                    raise ValueError(_arg + ' is required in config.')
                elif _arg == 'depth':
                    _config.depth = 1
                elif _arg == 'steps_per_epoch':
                    _config.steps_per_epoch = -1
                elif _arg == 'convert_to':
                    _config.convert_to = 'RGB'
                elif _arg == 'modcrop':
                    _config.modcrop = True
        self.depth = _config.depth
        self.patch_size = _config.patch_size
        self.scale = Utility.to_list(_config.scale, 2)
        self.patches_per_epoch = _config.steps_per_epoch * _config.batch
        self.batch = _config.batch
        self.crop = _config.crop
        self.modcrop = _config.modcrop
        self.resample = _config.resample
        return _config

    def _read_file(self, dataset):
        """Initialize all `File` objects"""
        if dataset.mode.lower() == 'pil-image1':
            if self.flow:
                # map flow
                flow = {f.stem: f for f in self.flow}
                self.file_objects = [ImageFile(fp).attach_flow(flow[fp.stem])
                                     for fp in self.file_names]
            else:
                self.file_objects = [ImageFile(fp) for fp in self.file_names]
        elif dataset.mode.upper() in _ALLOWED_RAW_FORMAT:
            self.file_objects = [
                RawFile(fp, dataset.mode, (dataset.width, dataset.height))
                for fp in self.file_names]
        elif dataset.mode.lower() == 'numpy':
            """already loaded numpy array, in case anyone want to use 
            external loaders, data can be a 4-D or 5-D ndarray"""
            tf.logging.debug('reading numpy array')
            param = dataset.numpy
            if param.exec is not None:
                exec(param.exec)
            src = param.get(self.method)
            if src:
                data = eval(param.get(self.method))
            else:
                data = None
            if isinstance(data, np.ndarray):
                for i, hr in enumerate(data):
                    if hr.ndim == 3:
                        frames_hr = [array_to_img(hr, 'RGB')]
                    else:
                        frames_hr = [array_to_img(x, 'RGB') for x in hr]
                    frames_lr = [imresize(img,
                                          np.reciprocal(self.scale,
                                                        dtype='float32'),
                                          resample=self.resample)
                                 for img in frames_hr]
                    frames_hr = [img.convert(self.color_format) for img in
                                 frames_hr]
                    frames_lr = [img.convert(self.color_format) for img in
                                 frames_lr]
                    name = (dataset.name, i, len(frames_hr))
                    self.frames.append((frames_hr, frames_lr, name))
            self.loaded = 1
            self.file_objects = []
        return self

    def _calc_select_prob(self, method=Select.EQUAL_PIXEL):
        """Get probability for selecting each file object.

        Args:
            method: We offer two method, see `Select` for details.
        """
        weights = []
        for f in self.file_objects:
            if method == Select.EQUAL_PIXEL:
                weights += [np.prod(f.shape) * f.frames]
            elif method == Select.EQUAL_FILE:
                weights += [1]
            else:
                raise ValueError('unknown select method ' + str(method))
        prob = np.array(weights, 'float32') / np.sum(weights, dtype='float32')
        prob = np.cumsum(prob)
        return prob

    def _random_select(self, size, seed=None):
        """Randomly select `size` file objects

        Args:
            size: the number of files to select
            seed: set the random seed (of `numpy.random`)

        Return:
            Dict: map file objects to its select quantity.
        """
        if seed:
            np.random.seed(seed)
        x = np.random.rand(size)
        # Q: Is `s` relevant to poisson dist.?
        s = {f: 0 for f in self.file_objects}
        for _x in x.tolist():
            _x *= np.ones_like(self.prob)
            diff = self.prob >= _x
            index = diff.nonzero()[0].tolist()
            if index:
                index = index[0]
            else:
                index = 0
            s[self.file_objects[index]] += 1
        return s

    def _vf_gen_lr_hr_pair(self, vf, depth, index):
        vf.seek(index)
        frames_hr = [shrink_to_multiple_scale(img, self.scale)
                     if self.modcrop else img for img in vf.read_frame(depth)]

        if self.scale == 1:
            frames_lr = [imcompress(img, random.randint(10, 60)) for img in frames_hr]
        else:
            frames_lr = [imresize(img,
                np.reciprocal(self.scale, dtype='float32'),
                resample=self.resample)
                for img in frames_hr]
        frames_hr = [img.convert(self.color_format) for img in frames_hr]
        frames_lr = [img.convert(self.color_format) for img in frames_lr]
        return frames_hr, frames_lr, (vf.name, index, vf.frames)

    def _vf_gen_flow_img_pair(self, vf, depth, index):
        assert depth == 2 and index == 0
        img = [img for img in vf.read_frame(depth)]
        img = [i.convert(self.color_format) for i in img]
        return img, [vf.flow], (vf.name, index, vf.frames)

    def _process_at_file(self, vf, clips=1):
        """load frames of `File` into memory, crop and generate corresponded
         LR frames.

        Args:
            vf: A `File` object.
            clips: an integer to specify how many clips to generate from `vf`.

        Return:
            List of Tuple: containing (HR, LR, name) respectively
        """
        assert isinstance(vf, (RawFile, ImageFile))

        tf.logging.debug('Prefetching ' + vf.name)
        depth = self.depth
        # read all frames if depth is set to -1
        if depth == -1:
            depth = vf.frames
        index = np.arange(0, vf.frames - depth + 1)
        np.random.shuffle(index)
        frames = []
        for i in index[:clips]:
            if self.flow:
                frames.append(self._vf_gen_flow_img_pair(vf, depth, i))
            else:
                frames.append(self._vf_gen_lr_hr_pair(vf, depth, i))
        vf.reopen()  # necessary, rewind the read pointer
        return frames

    def _generate_crop_grid(self, frames, size, shuffle=False):
        """generate randomly cropped box of `frames`

        Args:
            frames: a list of tuple, commonly returned from `_process_at_file`.
            size: an int scalar to specify number of generated crops.
            shuffle: a boolean, whether to shuffle the outputs.

        Return:
            list of tuple: containing (HR, LR, box, name) respectively,
              where HR and LR are reference frames, box is a list of 4
              int of crop coordinates.
        """
        if not frames:
            tf.logging.warning('frames is empty. [size={}]'.format(size))
            return []
        patch_size = Utility.to_list(self.patch_size, 2)
        patch_size = Utility.shrink_mod_scale(patch_size, self.scale)
        if size < 0:
            index = np.arange(len(frames)).tolist()
        else:
            if self.crop == 'random':
                index = np.random.randint(len(frames), size=size).tolist()
            else:
                index = np.arange(size).tolist()
        grids = []
        for i in range(len(frames)):
            hr, lr, name = frames[i]
            _w, _h = hr[0].width, hr[0].height
            if self.crop == 'not' or self.crop is None:
                _pw, _ph = _w, _h
            else:
                _pw, _ph = patch_size
            amount = index.count(i)
            if self.crop == 'random':
                x = np.random.randint(0, _w - _pw + 1, size=amount)
                y = np.random.randint(0, _h - _ph + 1, size=amount)
            elif self.crop == 'center':
                x = np.array([(_w - _pw) // 2] * amount)
                y = np.array([(_h - _ph) // 2] * amount)
            else:
                x = np.zeros([amount])
                y = np.zeros([amount])
            x -= x % self.scale[0]
            y -= y % self.scale[1]
            grids += [(hr, lr, [_x, _y, _x + _pw, _y + _ph], name)
                      for _x, _y in zip(x, y)]
        if shuffle:
            np.random.shuffle(grids)
        return grids

    @property
    def size(self):
        """expected total memory usage of the loader"""
        bpp = 3  # bytes per pixel
        if self.flow:
            bpp += 8  # two more float channel
        # NOTE use uint64 to prevent sum overflow
        return np.sum([np.prod((*vf.shape, vf.frames, bpp), dtype=np.uint64)
                       for vf in self.file_objects])

    def __len__(self):
        """length of a BasicLoader is defined as the total frames in Dataset"""
        return np.sum([vf.frames for vf in self.file_objects])

    def change_select_method(self, method):
        """change to different select method, see `Select`"""
        self.prob = self._calc_select_prob(method)
        return self

    def _prefetch(self, memory_usage=None, shard=1, index=0):
        """Prefetch `size` files and load into memory. Specify `shard` will
        divide loading files into `shard` shards in order to execute in
        parallel.
>>>>>>> cc34163... Generate compressed LR images at scale = 1

class QuickLoader(BasicLoader):
  """Async data loader with high efficiency.

  `QuickLoader` concurrently pre-fetches clips into memory every n iterations,
  and provides several methods to select clips. `QuickLoader` won't loads all
  files in the dataset into memory if your memory isn't enough.

  NOTE: A clip is a bunch of consecutive frames, which can represent either a
  dynamic video or single image.

  Args:
      dataset: A `Dataset` to load by this loader.
      method: A string in ('train', 'val', 'test') specifies which subset to
        use in the dataset. Also 'train' set will shuffle buffers each epoch.
      config: A `Config` class, including 'batch', 'depth', 'patch_size',
        'scale', 'steps_per_epoch' and 'convert_to' arguments.
      augmentation: A boolean to specify whether call `_augment` to batches.
        `_augment` will randomly flip or rotate images.
      n_threads: number of threads to load dataset
      kwargs: override config key-values.
  """

  def __init__(self, dataset, method, config, augmentation=False, n_threads=1,
               **kwargs):

    self.shard = n_threads
    self.threads = []
    super(QuickLoader, self).__init__(dataset, method, config,
                                      augmentation, **kwargs)

  def prefetch(self, memory_usage=None):
    """Prefetch data.

    This call will spawn threads of `_prefetch` and returns immediately.
    The next call of `make_one_shot_iterator` will join all the threads.
    If this is not called in advance, data will be fetched at
    `make_one_shot_iterator`.

    Args:
        memory_usage: desired virtual memory to use, could be int (bytes) or
          a readable string ('3GB', '1TB'). Default to use all available
          memories.

    Note: call `prefetch` twice w/o `make_one_shot_iterator` is
      undefined behaviour.
    """

    for i in range(self.shard):
      t = th.Thread(target=self._prefetch,
                    args=(memory_usage, self.shard, i),
                    name='fetch_thread_{}'.format(i))
      t.start()
      self.threads.append(t)

  def make_one_shot_iterator(self, memory_usage=None, shuffle=False):
    """make an `EpochIterator` to enumerate batches of the dataset. Specify
    `shard` will divide loading files into `shard` shards in order to
    execute in parallel.

    Will create TFIterator if use TFRecordDataset.

    Args:
        memory_usage: desired virtual memory to use, could be int (bytes) or
          a readable string ('3GB', '1TB'). Default to use all available
          memories.
        shuffle: A boolean whether to shuffle the patch grids.

    Return:
        An EpochIterator or TFIterator

    Known issues:
        If data of either shard is too large (i.e. use 1 shard and total
        frames is around 6GB in my machine), windows Pipe may broke and
        `get()` never returns.
    """

    if not self.threads:
      self.prefetch(memory_usage)
    for t in self.threads:
      t.join()
    self.threads.clear()
    # reduce
    grids = self._generate_crop_grid(self.frames,
                                     self.patches_per_epoch,
                                     shuffle=shuffle)
    if not (self.loaded & 0xFFFF):
      self.frames.clear()
    return EpochIterator(self, grids)
