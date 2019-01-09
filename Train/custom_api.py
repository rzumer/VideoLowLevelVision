"""
Copyright: Wenyi Tang 2017-2018
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: July 31st 2018

custom feature callback.
Usage:
    pass the function name to `run.py` args `--add_custom_callbacks func1 --add_custom_callbacks func2`.
    During training and testing, the `Trainer` will call the given functions to process input data

Note:
    functions must keep `kwargs` to ignore un-parsed arguments
"""

from PIL import Image
import numpy as np


def color_inverse(inputs, name, **kwargs):
    type_old = inputs.dtype
    inputs = inputs.astype('float')
    return np.clip(255 - inputs, 0, 255).astype(type_old)


def color_shift(inputs, name, **kwargs):
    type_old = inputs.dtype
    inputs = inputs.astype('float')
    return np.clip(inputs - 50, 0, 255).astype(type_old)


def red(inputs, **kwargs):
    x = np.zeros_like(inputs)
    x[..., 0] = 255
    return x


def green(inputs, **kwargs):
    x = np.zeros_like(inputs)
    x[..., 1] = 255
    return x


def blue(inputs, **kwargs):
    x = np.zeros_like(inputs)
    x[..., 2] = 255
    return x


def chessboard(inputs, **kwargs):
    x = np.zeros_like(inputs)
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i::64, j::64, :] = c
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i + 32::64, j::64, :] = c
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i::64, j + 32::64, :] = c
    c = np.random.randint(0, 255, [1, 1, 1, 3])
    for i in range(32):
        for j in range(32):
            x[:, i + 32::64, j + 32::64, :] = c
    return x


def noisy(inputs, sigma=15, **kwargs):
    shape = inputs.shape
    return np.random.normal(0, sigma, shape)


def add_noise(inputs, sigma=15, **kwargs):
    noise = noisy(inputs, sigma)
    noise += inputs.astype('float32')
    return np.clip(np.round(noise), 0, 255).astype('uint8')


def add_random_noise(inputs, sigma_max=50, **kwargs):
    sigma = np.random.randint(0, int(sigma_max))
    noise = np.random.normal(0, sigma, inputs.shape)
    noise += inputs.astype('float32')
    return np.clip(np.round(noise), 0, 255).astype('uint8')


def shave(inputs, div=64, **kwargs):
    h, w = inputs.shape[-3:-1]
    h_mod = h - h % div
    w_mod = w - w % div
    return inputs[..., :h_mod, :w_mod, :]


def pad(inputs, div=64, **kwargs):
    h, w = inputs.shape[-3:-1]
    ph = div - h % div
    pw = div - w % div
    if ph == div: ph = 0
    if pw == div: pw = 0
    ph = [ph // 2, ph - ph // 2]
    pw = [pw // 2, pw - pw // 2]
    inputs = np.pad(inputs, [[0, 0], [0, 0], ph, pw, [0, 0]], 'edge')
    return inputs


def upsample(inputs, scale=4, **kwargs):
    res = []
    for img in inputs:
        h, w, c = img.shape
        if c == 3:
            img = Image.fromarray(img, 'RGB')
        elif c == 1:
            img = Image.fromarray(img[..., 0], 'L')
        else:
            raise ValueError
        img = img.resize([w * scale, h * scale], resample=Image.BICUBIC)
        res.append(np.array(img))
    res = np.stack(res)
    if np.ndim(res) < 4:
        res = np.expand_dims(res, axis=-1)
    return res
