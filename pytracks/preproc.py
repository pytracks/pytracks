'''


Author: Alexandre Fioravante de Siqueira
Version: february, 2016
'''

from skimage import img_as_float
from skimage import exposure
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize_3d

import numpy as np


class RegionsAndSkel:
    """
    """

    def __init__(self, image_bin):
        self.img_bin = image_bin

        if np.asarray(image_bin).ndim == 2:
            image_bin = np.expand_dims(image_bin, axis=0)

    @staticmethod
    def _ret_imgprops(image_bin):
        image_props = []
        for img in image_bin:
            image_props.append(regionprops(label(img)))
        return image_props

    @staticmethod
    def _calc_skel(image_props):
        skeletons = []

        for props in image_props:
            skeletons.append([skeletonize_3d(prop.img) for prop in props])
        return skeletons

    @staticmethod
    def _ret_regions(image_props):
        regions = []
        for props in image_props:
            regions.append([prop.img for prop in props])
        return regions

    @property
    def skel(self):

        return self._calc_skel(self.image_props)

    @property
    def regions(self):

        return self._ret_regions(self.image_props)


def histogram_equalization(img, typeeq='histeq'):
    """
    Implements options for histogram equalization.
    Parameters
        img
        typeeq: 'histeq', 'adapteq', 'contstr'

    Returns
        img_eq
    """

    img = img_as_float(img)

    if typeeq is 'histeq':
        img_eq = exposure.equalize_hist(img)
    elif typeeq is 'adapteq':
        img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)
    elif typeeq is 'contstr':
        p2, p98 = np.percentile(img, (2, 98))
        img_eq = exposure.rescale_intensity(img, in_range=(p2, p98))

    return img_eq
