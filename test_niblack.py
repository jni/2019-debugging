# the code below will return NaN values only.
# depending on the image's pixel values,
# it's possible to get back a mix of NaN and float values

import numpy as np
from modules.niblack import threshold_niblack


image1 = np.zeros((5, 5))
image1[1:3, 1:3] = 0.5
image1[2, 2] = 0.7

print('\n\nimage1 image, threshold, thresholded')
print(image1)
print(threshold_niblack(image1, 3))
print(image1 > threshold_niblack(image1, 3))


image2 = np.full((4, 4), 0.03082192 + 2.19178082e-09)

print('\n\nimage2 image, threshold, thresholded')
print(image2)
print(threshold_niblack(image2))
print(image2 > threshold_niblack(image2))
