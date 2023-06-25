import numpy as np
import argparse

import matplotlib.pyplot as plt

import yaml
import rowan

from pathlib import Path

import torch
import cv2

def gen_transformation_matrix(sf, tx, ty):
    matrix = np.zeros((3,3))
    matrix[0,0] = sf
    matrix[1,1] = sf
    matrix[0, 2] = tx
    matrix[1, 2] = ty
    matrix[2, 2] = 1.
    return matrix

# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
def opencv2quat(rvec):
    angle = np.linalg.norm(rvec)
    if angle == 0:
        q = np.array([1,0,0,0])
    else:
        axis = rvec.flatten() / angle
        q = rowan.from_axis_angle(axis, angle)
    return q

def get_bb_patch(transformation):
    transformation_t = torch.tensor(transformation).unsqueeze(0)
    inv_transform = torch.inverse(transformation_t)[:, :2]
    affine_grid = torch.nn.functional.affine_grid(inv_transform, size=(1, 1, 96, 160), align_corners=True)

    transformed_patch = torch.nn.functional.grid_sample(torch.ones(1, 1, 96, 160).double(), affine_grid, align_corners=True, padding_mode='zeros')

    patch_coords = torch.nonzero(transformed_patch)[..., 2:]

    xmin = patch_coords[0][1].item()
    ymin = patch_coords[0][0].item()
    xmax = patch_coords[-1][1].item()
    ymax = patch_coords[-1][0].item()

    # fig, ax = plt.subplots()
    # ax.imshow(transformed_patch[0][0], cmap='gray')
    # plt.show()

    return [xmin, ymin, xmax, ymax]

# printed patch width = 28.2 cm
RADIUS = 0.1405 # in m

# compute relative position of center of patch in camera frame
def xyz_from_bb(bb,mtrx, dist_coeffs):
    # bb - xmin,ymin,xmax,ymax
    # mtrx, dist_vec = get_camera_parameters()
    fx = np.array(mtrx)[0][0]
    fy = np.array(mtrx)[1][1]
    ox = np.array(mtrx)[0][2]
    oy = np.array(mtrx)[1][2]
    # get pixels for bb side center
    P1 = np.array([bb[0],(bb[1] + bb[3])/2])
    P2 = np.array([bb[2],(bb[1] + bb[3])/2])
    # rectify pixels
    P1_rec = cv2.undistortPoints(P1, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix
    P2_rec = cv2.undistortPoints(P2, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix

    # get rays for pixels
    a1 = np.array([(P1_rec[0]-ox)/fx, (P1_rec[1]-oy)/fy, 1.0])
    a2 = np.array([(P2_rec[0]-ox)/fx, (P2_rec[1]-oy)/fy, 1.0])
    # normalize rays
    a1_norm = np.linalg.norm(a1)
    a2_norm = np.linalg.norm(a2)
    # get the distance    
    distance = (np.sqrt(2)*RADIUS)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))
    # get central ray
    ac = (a1+a2)/2
    # get the position
    xyz = distance*ac/np.linalg.norm(ac)
    return xyz



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path')


    args = parser.parse_args()

    patch = np.load(Path(args.path) / 'patches.npy')[-1][0][0]

    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.imshow(patch, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('print_patch.jpg', dpi=100)


    positions = np.load(Path(args.path) / 'positions_norm.npy')[:, -1] # positions is of shape (), hl_iterations, )

    positions = np.rollaxis(positions, 0, 3)

    print(positions)



    with open('misc/camera_calibration/calibration.yaml') as f:
        camera_config = yaml.load(f, Loader=yaml.FullLoader)

    camera_intrinsic = np.array(camera_config['camera_matrix'])
    distortion_coeffs = np.array(camera_config['dist_coeff'])
    
    rvec = np.array(camera_config['rvec'])
    tvec = camera_config['tvec']

    camera_extrinsic = np.zeros((4,4))
    camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
    camera_extrinsic[:3, 3] = tvec
    camera_extrinsic[-1, -1] = 1.

    
    T_patch_victim = np.zeros((positions.shape[0], positions.shape[1], 3))

    


    for k in range(positions.shape[0]):
        for m in range(positions.shape[1]):
            transformation_matrix = gen_transformation_matrix(*positions[m, k])
            
            bounding_box_placed_patch = get_bb_patch(transformation_matrix)

            patch_in_camera = xyz_from_bb(bounding_box_placed_patch, camera_intrinsic, distortion_coeffs)
            # print(patch_in_camera)
            patch_in_victim = (np.linalg.inv(camera_extrinsic) @ [*patch_in_camera, 1])[:3]

            T_patch_victim[m, k, :] = patch_in_victim

    print(T_patch_victim)

    dict = {'sf': [], 'tx': [], 'ty': []}

    dict['sf'] = T_patch_victim.T[0].tolist()
    dict['tx'] = T_patch_victim.T[1].tolist()
    dict['ty'] = T_patch_victim.T[2].tolist()

    print(dict)


    with open('T_patch_victim.yaml', 'w') as file:
        yaml.dump(dict, file)