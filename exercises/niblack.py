import itertools
import numpy as np
from scipy import ndimage as ndi
from collections.abc import Iterable
from skimage.transform import integral_image
from skimage.util import crop


def _validate_window_size(axis_sizes):
    """Ensure all sizes in ``axis_sizes`` are odd.

    Parameters
    ----------
    axis_sizes : iterable of int

    Raises
    ------
    ValueError
        If any given axis size is even.
    """
    for axis_size in axis_sizes:
        if axis_size % 2 == 0:
            msg = ('Window size for `threshold_sauvola` or '
                   '`threshold_niblack` must not be even on any dimension. '
                   'Got {}'.format(axis_sizes))
            raise ValueError(msg)


def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window size ``w``.
    The algorithm uses integral images to speedup computation. This is
    used by :func:`threshold_niblack` and :func:`threshold_sauvola`.

    Parameters
    ----------
    image : ndarray
        Input image.
    w : int, or iterable of int
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).

    Returns
    -------
    m : ndarray of float, same shape as ``image``
        Local mean of the image.
    s : ndarray of float, same shape as ``image``
        Local standard deviation of the image.

    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
           :DOI:`10.1117/12.767755`
    """
    if not isinstance(w, Iterable):
        w = (w,) * image.ndim
    _validate_window_size(w)

    pad_width = tuple((k // 2 + 1, k // 2) for k in w)
    padded = np.pad(image.astype('float'), pad_width,
                    mode='reflect')
    padded_sq = padded * padded

    integral = integral_image(padded)
    integral_sq = integral_image(padded_sq)

    kern = np.zeros(tuple(k + 1 for k in w))
    for indices in itertools.product(*([[0, -1]] * image.ndim)):
        kern[indices] = (-1) ** (image.ndim % 2 != np.sum(indices) % 2)

    total_window_size = np.prod(w)
    sum_full = ndi.correlate(integral, kern, mode='constant')
    m = crop(sum_full, pad_width) / total_window_size
    sum_sq_full = ndi.correlate(integral_sq, kern, mode='constant')
    g2 = crop(sum_sq_full, pad_width) / total_window_size
    # Note: we use np.clip because g2 is not guaranteed to be greater than
    # m*m when floating point error is considered
    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return m, s


def threshold_niblack(image, window_size=15, k=0.2):
    """Applies Niblack local threshold to an array.

    A threshold T is calculated for every pixel in the image using the
    following formula::

        T = m(x,y) - k * s(x,y)

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.

    Parameters
    ----------
    image: ndarray
        Input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of parameter k in threshold formula.

    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    References
    ----------
    .. [1] Niblack, W (1986), An introduction to Digital Image
           Processing, Prentice-Hall.

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> binary_image = threshold_niblack(image, window_size=7, k=0.1)
    """
    m, s = _mean_std(image, window_size)
    return m - k * s
