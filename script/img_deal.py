import pycapt
from PIL import Image
import numpy as np


im = Image.open("img/0236_1562399080.051659.jpg")

matrix = np.asarray(im)

print(matrix.shape)

im = im.convert("L")

matrix = np.asarray(im)

print(matrix.shape)






