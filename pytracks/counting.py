from collections import namedtuple
from skimage.measure import regionprops

import numpy as np


def count_by_region(img_regions):
    """
    """

    # 0: indexes; 1: region tracks; 2: total tracks
    count_auto = []
    RegionCount = namedtuple('AutoCount', 'idx reg_tracks')

    for idx, img in enumerate(img_regions[2]):
        aux_sum = 0
        trk_area, trk_pts = tracks_classify(img)

        try:
            aux_sum += len(trk_area)
        except TypeError:
            aux_sum += 1

        count_auto.append(RegionCount(idx=idx, reg_tracks=aux_sum))

    return count_auto


def count_tracks_area(labels, max_area=1.5):
    """
    """

    reg_area = []
    mean_area = 0
    num_obj = 0

    regions = regionprops(labels)
    for reg in regions:
        reg_area.append([reg.label, reg.area])
        mean_area += reg.area

    mean_area /= len(regions)
    for area in reg_area:
        area.append(np.trunc(area[1] / (mean_area * max_area)) + 1)

    return reg_area, mean_area
