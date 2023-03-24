'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''
import os
import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")

import shutil
import numpy as np
from subprocess import call

from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj
from scipy.io import loadmat, savemat

DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/weights.npy'


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(['curl', 'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy',
              '--create-dirs', '-o', fn])


def load_demo_images():
    ims = []
    # for i in range(1):
    #     im = Image.open('imgs/0%d.png' % i)
    #     ims.append([np.array(im).transpose(
    #         (2, 0, 1)).astype(np.float32) / 255.])
    im = Image.open('../imgs/shapeNet/05.png')
    ims.append([np.array(im).transpose(
    (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'
    voxel_file_name = sys.argv[2] if len(sys.argv) > 2 else 'out.npy'

    # load images
    demo_imgs = load_demo_images()

    # Download and load pretrained weights
    download_model(DEFAULT_WEIGHTS)

    # Use the default network model
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass(compute_grad=False)  # instantiate a network
    net.load(DEFAULT_WEIGHTS)                        # load downloaded weights
    solver = Solver(net)                # instantiate a solver

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)

    voxel_prediction = voxel_prediction[0,:,1,:,:]
    
    # original with pix3d
    
    # # for i in range(voxel_prediction.shape[1]):
    #     # voxel_prediction[:,i,:] = -1 * voxel_prediction[:,i,:]
    # # print("voxels shape,", voxel_prediction.shape)
    # # voxel_prediction = np.negative(voxel_prediction, where=[False, False, True])
    # # np.save(voxel_file_name, voxel_prediction[0, :, 1, :, :])
    # voxel_prediction = np.transpose(voxel_prediction, (2,1,0))
    # voxel_prediction = np.flip(voxel_prediction, axis=2)
    # np.save(voxel_file_name, voxel_prediction)
    # # Save the prediction to an OBJ file (mesh file).
    # # voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)
    
    voxel_prediction = np.transpose(voxel_prediction, (0,2,1))
    np.save(voxel_file_name, voxel_prediction)
    # voxel = loadmat("plane.mat")['voxel']

    voxel2obj(pred_file_name, voxel_prediction > cfg.TEST.VOXEL_THRESH)


    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    if cmd_exists('meshlab'):
        call(['meshlab', pred_file_name])
    else:
        print('Meshlab not found: please use visualization of your choice to view %s' %
              pred_file_name)


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    main()
