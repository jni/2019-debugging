import numpy as np


# Computing a histogram using np.histogram on a uint8 image with bins=256
# doesn't work and results in aliasing problems. We use a fully specified set
# of bins to ensure that each uint8 value false into its own bin.
_DEFAULT_ENTROPY_BINS = tuple(np.arange(-0.5, 255.51, 1))


def cross_entropy(image, threshold, bins=_DEFAULT_ENTROPY_BINS):
    """Compute cross-entropy between distributions above and below a threshold.

    Parameters
    ----------
    image : array
        The input array of values.
    threshold : float
        The value dividing the foreground and background in ``image``.
    bins : int or array of float, optional
        The number of bins or the bin edges. (Any valid value to the ``bins``
        argument of ``np.histogram`` will work here.) For an exact calculation,
        each unique value should have its own bin. The default value for bins
        ensures exact handling of uint8 images: ``bins=256`` results in
        aliasing problems due to bin width not being equal to 1.

    Returns
    -------
    nu : float
        The cross-entropy target value as defined in [1]_.

    Notes
    -----
    See Li and Lee, 1993 [1]_; this is the objective function `threshold_li`
    minimizes. This function can be improved but this implementation most
    closely matches equation 8 in [1]_ and equations 1-3 in [2]_.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           :DOI:`10.1016/S0167-8655(98)00057-9`
    """
    histogram, bin_edges = np.histogram(image, bins=bins, density=True)
    bin_centers = np.convolve(bin_edges, [0.5, 0.5], mode='valid')
    t = np.flatnonzero(bin_centers > threshold)[0]
    m0a = np.sum(histogram[:t])  # 0th moment, background
    m0b = np.sum(histogram[t:])
    m1a = np.sum(histogram[:t] * bin_centers[:t])  # 1st moment, background
    m1b = np.sum(histogram[t:] * bin_centers[t:])
    mua = m1a / m0a  # mean value, background
    mub = m1b / m0b
    nu = -m1a * np.log(mua) - m1b * np.log(mub)
    return nu


def threshold_li(image, *, tolerance=None):
    """Compute threshold value by Li's iterative Minimum Cross Entropy method.

    Parameters
    ----------
    image : ndarray
        Input image.

    tolerance : float, optional
        Finish the computation when the change in the threshold in an iteration
        is less than this value. By default, this is half of the range of the
        input image, divided by 256.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           :DOI:`10.1016/S0167-8655(98)00057-9`
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           :DOI:`10.1117/1.1631315`
    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_li(image)
    >>> binary = image > thresh
    """
    image_range = np.max(image) - np.min(image)
    tolerance = tolerance or 0.5 * image_range / 256

    # Initial estimate
    t_curr = np.mean(image)
    t_next = t_curr + 2 * tolerance

    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance
    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = (image > t_curr)
        mean_fore = np.mean(image[foreground])
        mean_back = np.mean(image[~foreground])

        t_next = ((mean_back - mean_fore) /
                  (np.log(mean_back) - np.log(mean_fore)))

    threshold = t_next
    return threshold

