"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Last update date: Mar. 25th 2019
"""

#  Copyright (c): Wenyi Tang 2017-2019.
#  Author: Wenyi Tang
#  Email: wenyi.tang@intel.com
#  Update Date: 2019/4/3 下午5:03

from setuptools import find_packages
from setuptools import setup

VERSION = '0.1.0'

REQUIRED_PACKAGES = [
  'numpy',
  'scipy',
  'matplotlib',
  'Pillow',
  'pypng',
  'pytest',
  'PyYAML',
  'psutil',
  'tqdm',
  'h5py',
  'easydict >= 1.9',
  'google-api-python-client',
  'oauth2client',
]

try:
  import torch

  # tensorboardX is absorbed into 1.1.0 now
  REQUIRED_PACKAGES.extend([
    'torch >= 1.1.0',
    'torchvision',
  ])
except ImportError:
  pass

setup(
  name='VLLV',
  version=VERSION,
  description='Video Low Level Vision Framework',
  url='https://github.com/rzumer/VideoLowLevelVision',
  packages=find_packages(),
  install_requires=REQUIRED_PACKAGES,
  license='MIT',
  author='Wenyi Tang, Raphaël Zumer',
  author_email='wenyitang@outlook.com, rzumer@tebako.net',
  keywords="super-resolution sr vsr tensorflow pytorch",
)
