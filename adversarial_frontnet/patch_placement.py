import numpy as np
import torch
from rowan import to_matrix


def calc_T_in_attaker_frame(patch_size, scale_factor=0.01, xyz_translations=[0., 0., 0.]):
    """
    Calculate the matrix for translating the path into the attacker UAV frame.
    Parameters:
        ----------
        patch_size: list, [height, width] of the patch
        scale_factor: float, set the scaling factor of the patch 
        xyz_translation: list, add a translation in [x, y, z]-direction
    Returns:
        a (4,4) numpy array, the transformation matrix
    """
    # the center of the patch will first be set to the center of the attacker
    # therefore, the upper left corner is shifted from (0,0) to (-height/2, -width/2)
    t_x = - patch_size[0] / 2
    t_y = - patch_size[1] / 2
    # no translation in z direction needed currently
    t_z = 0

    # adjust translation with scaling factor
    t_x *= scale_factor
    t_y *= scale_factor
    t_z *= scale_factor

    # now add the translations in x, y and z direction according 
    # to the exact placement of the patch on the attacker
    t_x = t_x + xyz_translations[0]
    t_y = t_y + xyz_translations[1]
    t_z = t_z + xyz_translations[2]

    # TODO: add calculation of rotation
    # for now, no rotation is performed


    T_patch_in_attacker = np.array([[scale_factor*1., 0., 0., t_x],
                                    [0., scale_factor*1., 0., t_y],
                                    [0., 0., scale_factor*1., t_z],
                                    [0., 0., 0., 1.]])
    
    return T_patch_in_attacker

def calc_T_attacker_in_camera(attacker_xyz, attacker_quaternions):
    """
    Calulate the matrix translating the 3D coordinates of the attacker UAV
    into camera frame given the quaternions.
    Parameters:
        ----------
        attacker_xyz: a (3,) numpy array, the 3D coordinates of the attacker UAV
        attacker_quaternions: a (4,) numpy array, the quaternions (qx, qy, qz, qw) 
                              describing the rotation of the attacker UAV to the camera
    Returns:
        a (4,4) numpy array, the transformation matrix
    """
    T_attacker_in_camera = np.zeros((4,4))
    # calculate rotation matrix from quaternions with rowan's to_matrix
    rotation = to_matrix(attacker_quaternions)
    # fill empty matrix with values for rotation and translation 
    T_attacker_in_camera[:3, :3] = rotation
    T_attacker_in_camera[:, 3][:3] = attacker_xyz.T
    T_attacker_in_camera[3, 3] = 1.
    
    return T_attacker_in_camera

def project_coords_to_image(patch_size, camera_config, T_attacker_in_camera, T_patch_in_attacker):
    """
    Function for calculating the image 2D coordinates of the whole patch.
    Has the same functionality as cv2.projectPoints().
    Parameters:
        ----------
        patch_size: list, [height, width] of the patch
        camera_config: dict, includes the camera intrinsics, translation matrix and distortion coefficients
        T_attacker_in_camera: a (4,4) numpy array, the calculated matrix translating the attacker UAV in camera frame
        T_patch_in_attacker: a (4,4) numpy array, the calculated matrix translating the patch in attacker UAV frame
    Returns:
        img_x: a (patch_height, patch_width) numpy array, including all projected x coordinates of the patch
        img_y: a (patch_height, patch_width) numpy array, including all projected y coordinates of the patch
    """
    # get a (4, n) matrix with all pixel coordinates of the patch
    indy, indx = np.indices((patch_size[0], patch_size[1]), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.zeros_like(indx).ravel(), np.ones_like(indx).ravel()])
   
    # transform coordinates to camera frame
    coords_in_camera = T_attacker_in_camera @ T_patch_in_attacker @ lin_homg_ind

    # convert camera config to numpy arrays
    camera_matrix = np.array(camera_config['camera_matrix'])
    translation_marix = np.array(camera_config['translation_matrix'])
    dist_coeffs = np.array(camera_config['dist_coeffs'])

    # store all distortion coeffecients in seperate variables
    k_1, k_2, p_1, p_2, k_3 = dist_coeffs

    # first: rotate pixel coordinates in camera frame into image frame
    coords_in_image = translation_marix @ coords_in_camera
    
    # second: consider distortion
    coords_dist = np.ones((coords_in_image.T.shape))
    for i, coords in enumerate(coords_in_image.T):
        x_ = coords[0] / coords[2]
        y_ = coords[1] / coords[2]

        r = np.sqrt(x_**2 + y_**2)

        x_d = x_ * (1+k_1*r**2+k_2*r**4+k_3*r**6) + 2*p_1*x_*y_+p_2
        y_d = y_ * (1+k_1*r**2+k_2*r**4+k_3*r**6) + p_1*(r**2+2*y_**2)+2*p_2*x_*y_

        coords_dist[i][0] = x_d
        coords_dist[i][1] = y_d


    # at last, transform into image pixel coordinates
    u, v, z = camera_matrix @ coords_dist.T
    # u and v need to be devided by z
    # simoultaneously round both arrays and store the values as int (floats are not needed here)
    img_x = np.round(u/z, decimals=0).astype(int)
    img_y = np.round(v/z, decimals=0).astype(int)

    # reshaping for easier use with following for loops
    img_x = img_x.reshape(patch_size[0], patch_size[1])
    img_y = img_y.reshape(patch_size[0], patch_size[1])
    return img_x, img_y
    
def get_bit_mask(image_coords, image_size):
    """"
    Calculate a bit mask for replacing the pixels of the transformed patch in the original image.
    Parameters:
        ----------
        image_coords: list of numpy arrays [img_x, img_y], the projected 2D pixel coordinates of the patch
        image_size: list, [height, width] of the original image
    Returns:
        a (height, width) numpy array, the bit mask for placing the patch
    """
    bit_mask = np.ones((image_size[0], image_size[1]))
    img_x, img_y = image_coords

    for x in range(img_x.shape[0]):
        for y in range(img_y.shape[0]):
            # only replace pixels that are actually visible in the image
            # this means we only replace pixels starting from the upper left corner (0,0)
            # until the lower right corner (height, width) of the original image
            # any pixels outside of the original image are ignored
            if img_x[x][y] >= 0. and img_y[x][y] >= 0.: 
                if img_x[x][y] < image_size[0] and img_y[x][y] < image_size[1]:
                    bit_mask[img_x[x][y]][img_y[x][y]] = 0.

    
    return bit_mask

def get_transformed_patch(image_coords, patch, image_size):
    """"
    Place all pixel values of the patch at the correct calculated position in a black image
    (for easier addition to the original image).
    Parameters:
        ----------
        image_coords: list of numpy arrays [img_x, img_y], the projected 2D pixel coordinates of the patch
        patch: a numpy array, the current iteration of the patch
        image_size: list, [height, width] of the original image
    Returns:
        a (height, width) numpy array, the placed patch in an otherwise black image
    """
    transformed_patch = np.zeros((image_size[0], image_size[1]))
    img_x, img_y = image_coords
    for x in range(img_x.shape[0]):
        for y in range(img_y.shape[1]):
            # only replace pixels that are actually visible in the image
            # this means we only replace pixels starting from the upper left corner (0,0)
            # until the lower right corner (height, width) of the original image
            # any pixels outside of the original image are ignored
            if img_x[x][y] >= 0. and img_x[x][y] >= 0.:
                if img_x[x][y] < image_size[0] and img_y[x][y] < image_size[1]:
                    transformed_patch[img_x[x][y]][img_y[x][y]] = patch[x][y]

    return transformed_patch
    

def place_patch(image, patch, attacker_pose, camera_config):
    """"
    Place all pixel values of the patch at the correct calculated position in the original image.
    Parameters:
        ----------
        image: a numpy array, the original image
        patch: a numpy array, the current iteration of the patch
        attacker_pose: a (7,) numpy array, including the 3D coordinates of the attacker UAV and the quaternions
        camera_config: dict, includes the camera intrinsics, translation matrix and distortion coefficients
    Returns:
        a numpy array, the final manipulated image including the placed patch
    """
    patch_size = [*patch.shape]
    image_size = [*image.shape]

    T_patch_in_attacker = calc_T_in_attaker_frame(patch_size=patch_size)
    
    T_attacker_in_camera = calc_T_attacker_in_camera(attacker_pose[:3], attacker_pose[3:])

    image_coords = project_coords_to_image(patch_size=patch_size, camera_config=camera_config, 
                   T_attacker_in_camera=T_attacker_in_camera, T_patch_in_attacker=T_patch_in_attacker)

    bit_mask = get_bit_mask(image_coords, image_size)
    transformed_patch = get_transformed_patch(image_coords, patch, image_size)

    image *= bit_mask
    image += transformed_patch

    return image

if __name__=="__main__":
    # ---Example for patch placement---
    # load the Frontnet dataset
    from util import load_dataset
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=False, drop_last=True, num_workers=0)

    # choose an image from the dataset
    # here, we chose the first pair (image, pose)
    image, pose = dataset.dataset.__getitem__(0)

    # the image is in shape (color channels, height, width)
    # since the images are grayscale, the color channel == 1
    image = image[0]
    # the pixel values of the images range from 0 to 255

    # pose is stored as x, y, z, phi -> phi is not needed for patch placement
    pose = pose[:3]

    # load the camera parameters from the config yaml file
    import yaml
    with open('adversarial_frontnet/camera_calibration/camera_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # and convert to numpy arrays
    camera_matrix = np.array(config['camera_matrix'])
    translation_marix = np.array(config['translation_matrix'])
    dist_coeffs = np.array(config['dist_coeffs'])

    # generate a random patch
    # first, generate random values between 0 and 1, 
    # then multiply by 255. to receive values between 0. and 255.
    patch = np.random.rand(100, 100) * 255.


    # set an arbitrary pose for testing
    # pose includes x,y,z and quaternions in order qx, qy, qz, qw
    # x,y,z will be the center of the patch in camera frame
    # quaternions hold rotation information from attacker (holding the patch) to camera frame
    pose = np.array([2.7143893241882324,1.6456797122955322,0.4578791558742523, 
                     0.0114797880217894, 0.0744068142306854, -0.1520472288581698, 0.985501639095322])

    # place the patch
    new_image = place_patch(image, patch, pose, config)
    # plot the final image
    import matplotlib.pyplot as plt
    plt.imshow(new_image)
    plt.show()
