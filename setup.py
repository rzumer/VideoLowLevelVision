from setuptools import find_packages
from setuptools import setup

VERSION = '0.1.0'

REQUIRED_PACKAGES = [
    'numpy',
    'Image',
    'pypng',
    'pytest',
    'PyYAML',
    'psutil',
    'tqdm',
    'easydict >= 1.9',
]

if __name__ == '__main__':
    setup(
        name='VLLV',
        version=VERSION,
        description='Video Low Level Vision Framework',
        url='https://github.com/rzumer/VideoLowLevelVision',
        packages=find_packages(),
        install_requires=REQUIRED_PACKAGES,
        license='MIT',
        author='Wenyi Tang, Raphaël Zumer',
        author_email='wenyitang@outlook.com, rzumer@tebako.net'
    )
