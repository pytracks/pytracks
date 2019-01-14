"""
"""

from itertools import product
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage.morphology import (binary_fill_holes,
                                      distance_transform_edt)
from scipy.stats import norm
from skimage import morphology
from skimage.color import gray2rgb
from skimage.io import ImageCollection, imread, imread_collection
from skimage.filters import threshold_isodata
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte
from sys import platform

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ['segmentation_wusem']


def segmentation_wusem(image, str_el='disk', initial_radius=10,
                       delta_radius=5):
    """Separates regions on a binary input image using successive
    erosions as markers for the watershed algorithm. The algorithm stops
    when the erosion image does not have objects anymore.

    Parameters
    ----------
    image : (N, M) ndarray
        Binary input image.
    str_el : string, optional
        Structuring element used to erode the input image. Accepts the
        strings 'diamond', 'disk' and 'square'. Default is 'disk'.
    initial_radius : int, optional
        Initial radius of the structuring element to be used in the
        erosion. Default is 10.
    delta_radius : int, optional
        Delta radius used in the iterations:
         * Iteration #1: radius = initial_radius + delta_radius
         * Iteration #2: radius = initial_radius + 2 * delta_radius,
        and so on. Default is 5.

    Returns
    -------
    img_labels : (N, M) ndarray
        Labeled image presenting the regions segmented from the input
        image.
    num_objects : int
        Number of objects in the input image.
    last_radius : int
        Radius size of the last structuring element used on the erosion.

    References
    ----------
    .. [1] F.M. Schaller et al. "Tomographic analysis of jammed ellipsoid
    packings", in: AIP Conference Proceedings, 2013, 1542: 377-380. DOI:
    10.1063/1.4811946.

    Examples
    --------
    >>> from skimage.data import binary_blobs
    >>> image = binary_blobs(length=512, seed=0)
    >>> img_labels, num_objects, _ = segmentation_wusem(image,
                                                        str_el='disk',
                                                        initial_radius=10,
                                                        delta_radius=3)
    """

    rows, cols = image.shape
    img_labels = np.zeros((rows, cols))
    curr_radius = initial_radius
    distance = distance_transform_edt(image)

    while True:
        aux_se = {
            'diamond': morphology.diamond(curr_radius),
            'disk': morphology.disk(curr_radius),
            'square': morphology.square(curr_radius)
        }
        str_el = aux_se.get('disk', morphology.disk(curr_radius))

        erod_aux = morphology.binary_erosion(image, selem=str_el)
        if erod_aux.min() == erod_aux.max():
            last_step = curr_radius
            break

        markers = label(erod_aux)
        curr_labels = morphology.watershed(-distance,
                                           markers,
                                           mask=image)
        img_labels += curr_labels

        # preparing for another loop.
        curr_radius += delta_radius

    # reordering labels.
    img_labels = label(img_labels)

    # removing small labels.
    img_labels, num_objects = label(remove_small_objects(img_labels),
                                    return_num=True)

    return img_labels, num_objects, last_step
