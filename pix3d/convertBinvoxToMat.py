from binvox_rw import read_as_3d_array
from scipy.io import savemat
import numpy as np
import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    outputname = sys.argv[2]
    model = read_as_3d_array(open(filename, 'rb'))
    voxels = model.data.astype(int).transpose(0,2,1)
    voxels[:, 1, :] = -voxels[:, 1, :]
    savemat(outputname, {'voxel': voxels})

# from meshrcnn.utils import shape as shape_utils
# my_voxels = shape_utils.read_voxel(mat_fname)
# https://github.com/xingyuansun/pix3d/issues/19