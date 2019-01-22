import numpy as np
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi

def scale_and_uint8(image, factor):
    """Interpolate an image by a given factor and convert the result to uint8.

    Parameters
    ----------
    image : array
    """
    coords = np.meshgrid(*(np.linspace(0, i, i * factor, endpoint=False)
                           for i in image.shape), indexing='ij')
    interpolated = ndi.map_coordinates(image, coords, mode='reflect')
    output = img_as_ubyte(interpolated)
    return output
