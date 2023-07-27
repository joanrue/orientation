import numpy as np
from pycsou.operator import StructureTensor
from eigh import eigh
from lode import lode_coords
import tifffile
import matplotlib.pyplot as plt
y = tifffile.imread("data/membrane.tif").astype("float32") / 255
#y = skimage.data.cells3d()[:,0]
# z_slice = 30
# plt.imshow(y[z_slice]), plt.show()

st = StructureTensor(arg_shape=y.shape)
x = st(y.ravel())
x = st.unravel(x)
# for i in range(6):
#      tifffile.imwrite(f"data/membrane_st{i}.tif", (x[i] * 255).astype("<u1"))

x = x.transpose(1, 2, 3, 0)
x2 = np.zeros((*x.shape[:-1], 9))
x2[..., [0, 1, 2, 4, 5, 8]] = x
w, v = eigh(x2.reshape(-1, 9), arg_shape=(3, 3))
w = w.reshape(*y.shape, 3)
# for i in range(3):
#      tifffile.imwrite(f"data/membrane_w{i}.tif", (((w[..., i] - w[..., i].min()) / (w[..., i] - w[..., i].min()).max()) * 255).astype("<u1"))

lode = lode_coords(v.reshape(-1, 3, 3))
lode = lode.reshape(*y.shape, 3)

for i in range(3):
     # tifffile.imwrite(f"data/membrane_lode{i}.tif", (lode[..., i] * 255).astype("<u1"))
     tifffile.imwrite(f"data/membrane_lode{i}.tif", lode[..., i])
     tifffile.imwrite(f"data/membrane_lode{i}_norm.tif", (((lode[..., i] - np.nanmin(lode[..., i])) / np.nanmax((lode[..., i] - np.nanmin(lode[..., i])))) * 255).astype("<u1"))
print("Done!")