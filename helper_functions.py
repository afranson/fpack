#!/usr/bin/env python3

import numpy as np


def _x_to_index(x_data, x_value, i=0):
    if len(x_data) == 1:
        return i
    half = len(x_data) // 2
    x_half = x_data[half]
    if x_half == x_value:
        return i
    elif len(x_data) < 8:
        diff = np.inf
        result = 0
        for n, x in enumerate(x_data):
            if abs(x - x_value) < diff:
                result = n
                diff = diff
        return i + result
    elif x_value < x_half:
        return _x_to_index(x_data[:half], x_value, i)
    return _x_to_index(x_data[half:], x_value, i + half)


def _xbaseline_to_ibaseline(x_data, xbaseline):
    if xbaseline[0] is None:
        xbaseline = [x_data[0], xbaseline[1]]
    if xbaseline[1] is None:
        xbaseline = [xbaseline[0], x_data[-1]]
    return (
        _x_to_index(x_data, xbaseline[0]),
        _x_to_index(x_data, xbaseline[1]),
    )


def _get_smoothing(y, smoothing):
    if smoothing is not None:
        return smoothing
    return len(y)


def _get_cut(x, y, xbaseline, cut_scale):
    if xbaseline != (None, None) or cut_scale is not None:
        if cut_scale is None:
            cut_scale = 2
        ibaseline = _xbaseline_to_ibaseline(x, xbaseline)
        y_mean = np.average(y[ibaseline[0] : ibaseline[1]])
        y_zeroed = y - y_mean
        cut = cut_scale * np.std(y_zeroed[ibaseline[0] : ibaseline[1]])
        cut_range = np.array([y_mean - cut, y_mean + cut])
        return cut_range

    n, bins = np.histogram(y)
    biggest_bin = np.argmax(n)
    cut_range = bins[biggest_bin : biggest_bin + 2]
    return cut_range


def get_scalebar_prop_width(
    scale_unit_w, *, img=None, img_pixel_w=1, img_unit_w=1, img_prop_w=1
):
    """Determines the proportional and pixel width of a scale bar that is
    'scale_unit_w' wide. Takes a PIL Image from
    Image.open(filename). Takes the width of the image in pixels and
    units (img_pixel_w and img_unit_w) to get to the pixel/unit ratio.
    To get the correct proportion, the function also needs the
    proportion of the figure that the image will take up, img_prop_w.
    return scale_prop_width
    """
    pixels_per_unit = img_pixel_w / img_unit_w
    scale_pixel_w = scale_unit_w * pixels_per_unit
    scale_prop_w = scale_pixel_w / img.size[0] * img_prop_w
    return scale_prop_w


def size_image_to_ratio(
    image, ratio, centers, pixel_width=None, pixel_height=None
):
    """Returns a cropped image centered at 'centers', 'pixel_width' wide,
    and with the appropriate ratio of width to height.
    """
    if pixel_width:
        pixel_height = ratio * pixel_width
    elif pixel_height:
        pixel_width = pixel_height / ratio
    else:
        raise ValueError(
            "Either pixel_width or pixel_height must be set."
            "Currently their values are {} and {}.".format(
                pixel_width, pixel_height
            )  # noqa
        )
    crop_widths = (centers[0] - pixel_width / 2, centers[0] + pixel_width / 2)
    crop_heights = (
        centers[1] - pixel_height / 2,
        centers[1] + pixel_height / 2,
    )  # noqa
    image = image.crop(
        (crop_widths[0], crop_heights[0], crop_widths[1], crop_heights[1])
    )
    return image
