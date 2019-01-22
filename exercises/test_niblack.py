# the code below will return NaN values only.
# depending on the image's pixel values,
# it's possible to get back a mix of NaN and float values

import numpy as np
from niblack import threshold_niblack

# 16 pixels, single channel
src = np.array([0.03082192 + 2.19178082e-09] * 16).astype('float64')
image = src.reshape((4,4))

print(threshold_niblack(image))
