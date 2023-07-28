import numpy as np
from pycsou.operator import StructureTensor
from eigh import eigh
from lode import lode_coords
import tifffile
import matplotlib.pyplot as plt

# Define input path and parameters
input_path = "data/SlicesY_bin4.tif"
sampling = (1., 1., 1.) # Change to correct values, are they isotropic voxels?
sigma = 2.5 # for structure tensor

# Load image
image = tifffile.imread(input_path).astype("float32")
image /= image.max()

# Define JIT-compiled structure tensor operator
structens = StructureTensor(arg_shape=image.shape,
                            smooth_sigma=sigma,
                            sampling=sampling)

# Apply structure tensor operator
st = structens.unravel(structens(image.ravel()))
st = st.transpose(1, 2, 3, 0)
# Reshape for Lode coordinate computations
st2 = np.zeros((*st.shape[:-1], 9))
st2[..., [0, 1, 2, 4, 5, 8]] = st

# Compute Lode coordinates
lode = lode_coords(st2.reshape(-1, 3, 3))
lode = lode.reshape(*image.shape, 3)

# Save result
for i in range(3):
     tifffile.imwrite(f"data/SliceY_lode{i}.tif", lode[..., i])
     tifffile.imwrite(f"data/SliceY_lode{i}_norm.tif", (((lode[..., i] - np.nanmin(lode[..., i])) / np.nanmax(
          (lode[..., i] - np.nanmin(lode[..., i])))) * 255).astype("<u1"))
print("Done!")