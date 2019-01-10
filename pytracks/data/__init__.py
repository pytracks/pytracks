import os

from os import path
from skimage import io

# defining the root data folder.
data_dir = path.abspath(path.dirname(__file__))

__all__ = ['dap01',
           'data_dir',
           'load']


def load(image_name, as_gray=False):
    """Load an image file located in the data directory.

    Parameters
    ----------
    image_name : string
        File name.
    as_gray : bool, optional (default : False)
        Convert to grayscale.

    Returns
    -------
    image : ndarray
        Image loaded from ``pytracks.data_dir''.
    """
    io.use_plugin('pil')
    return io.imread(path.join(data_dir, image_name), as_gray=as_gray)


def dap01():
    """DAP test image #01."""
    return load('dap01.bmp', as_gray=True)
