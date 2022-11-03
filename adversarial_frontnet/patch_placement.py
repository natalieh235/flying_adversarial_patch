import numpy as np
import torch
#from rowan import to_matrix
from torch.nn.functional import grid_sample


import matplotlib.pyplot as plt

def to_rotation_matrix(q, require_unit=True):
    # copy of rowan's to_matrix() function and adapting it for torch
    # source: https://github.com/glotzerlab/rowan/blob/1b64ac7399e86459ee95e8499b11919b83a30305/rowan/functions.py#L952
    s = torch.linalg.norm(q)
    if torch.any(s == 0.0):
        raise ZeroDivisionError("At least one element of q has approximately zero norm")
    elif require_unit and not torch.allclose(s, torch.tensor(1.0)):
        raise ValueError(
            "Not all quaternions in q are unit quaternions. \
If this was intentional, please set require_unit to False when \
calling this function."
        )

    m = torch.empty(q.shape[:-1] + (3, 3))
    s = torch.pow(s, -1.0)  # For consistency with Wikipedia notation
    m[..., 0, 0] = 1.0 - 2 * s * (q[..., 2] ** 2 + q[..., 3] ** 2)
    m[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    m[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    m[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    m[..., 1, 1] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 3] ** 2)
    m[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    m[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    m[..., 2, 1] = 2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
    m[..., 2, 2] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return m


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


    T_patch_in_attacker = torch.tensor([[scale_factor*1., 0., 0., t_x],
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
    T_attacker_in_camera = torch.zeros((4,4))
    # calculate rotation matrix from quaternions with rowan's to_matrix
    rotation = to_rotation_matrix(attacker_quaternions)
    # fill empty matrix with values for rotation and translation 
    T_attacker_in_camera[:3, :3] = rotation
    T_attacker_in_camera[:, 3][:3] = attacker_xyz
    T_attacker_in_camera[3, 3] = 1.
    
    return T_attacker_in_camera

def affine_grid(patch_size, image_size, camera_config, T_attacker_in_camera, T_patch_in_attacker):
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

    oh, ow = image_size[2:]
    h, w = patch_size[2:]

    base_grid = torch.empty(4, oh, ow)
    x_grid = torch.linspace(-1, 1, steps=ow)
    x_grid = x_grid * (ow - 1.) / ow
    base_grid[0].copy_(x_grid)
    y_grid = torch.linspace(-1, 1, steps=oh).unsqueeze_(-1)
    y_grid = y_grid * (oh - 1.) / oh
    base_grid[1].copy_(y_grid)
    base_grid[2].fill_(0)
    base_grid[3].fill_(1)
    base_grid = base_grid.view(4, -1)
    print(base_grid.shape)

    # transform coordinates to camera frame

    # coords_in_camera = T_attacker_in_camera @ T_patch_in_attacker @ base_grid


    # code not working with distortion coefficients
    # # convert camera config to numpy arraysase_grid#
    # camera_matrix = torch.tensor(camera_config['camera_matrix'])
    # translation_marix = torch.tensor(camera_config['translation_matrix'])
    # dist_coeffs = torch.tensor(camera_config['dist_coeffs'])

    # # # store all distortion coeffecients in seperate variables
    # k_1, k_2, p_1, p_2, k_3 = dist_coeffs

    # # first: rotate pixel coordinates in camera frame into image frame
    # coords_in_image = translation_marix @ coords_in_camera
    
    # # # second: consider distortion
    # coords_dist = torch.ones_like((coords_in_image.mT))
    # for i, coords in enumerate(coords_in_image.mT):
    #     x_ = coords[0] / coords[2]
    #     y_ = coords[1] / coords[2]

    #     r = torch.sqrt(x_**2 + y_**2)

    #     x_d = x_ * (1+k_1*r**2+k_2*r**4+k_3*r**6) + 2*p_1*x_*y_+p_2
    #     y_d = y_ * (1+k_1*r**2+k_2*r**4+k_3*r**6) + p_1*(r**2+2*y_**2)+2*p_2*x_*y_

    #     coords_dist[i][0] = x_d
    #     coords_dist[i][1] = y_d

    # # at last, transform into image pixel coordinates
    # u, v, z = camera_matrix.mT @ coords_dist.mT
    # # u and v need to be devided by z
    # img_x = u/z
    # img_y = v/z

    # 1x3x4 camera intrinsics from DLT calibration, 
    # using it for debugging affine grid and grid sample
    # camera_matrix = torch.tensor([[[  26.33732709,  -25.19487877, -114.26821427,   33.71230283],
    #                                     [  11.99457808,    4.20581382, -181.29537939,   57.02665397],
    #                                     [   0.38332883,   -0.00340662,   -3.07207869,    1.        ]]])

    #T_patch_in_camera = T_attacker_in_camera @ T_patch_in_attacker

    transformation_matrix = camera_matrix @ T_patch_in_camera @ T_base_grid

    coords = transformation_matrix @ base_grid
    print(coords, coords.shape)
    
    
    affine_grid = torch.stack([img_y, img_x]).mT
    affine_grid = affine_grid.reshape((1, oh, ow, 2))

    print(affine_grid.shape)

    return affine_grid
    
def get_bit_mask(patch_size, image_size, grid):
    """"
    Calculate a bit mask for replacing the pixels of the transformed patch in the original image.
    Parameters:
        ----------
        image_coords: list of numpy arrays [img_x, img_y], the projected 2D pixel coordinates of the patch
        image_size: list, [height, width] of the original image
    Returns:
        a (height, width) numpy array, the bit mask for placing the patch
    """
    bit_mask = torch.ones((image_size[2:]))

    #ori_x = torch.linspace(0, patch_size[2])
    #ori_y = torch.linspace(0, patch_size[3])


    # for x in range(image_size[2]):
    #     for y in range(image_size[3]):
    #         if x,y in grid:
    #             bit_mask[x,y] = patch[ori_x[x], ori_y[y]]

    # for k in range(patch_size[2]):
    #     for j in range(patch_size[3]):
    #         print(max(0, 1-torch.abs(grid[0]-j)))
    #         print(max(0, 1-torch.abs(grid[1]-k)))
    


    # for x in range(image_size[2]):
    #     for y in range(image_size[3]):
    #         bit_mask[x,y] = 
    # for x in range(image_size[2]):
    #     for y in range(image_size[3]):


    #img_x, img_y = grid.reshape(2, 100, 100)

    # for x in range(img_x.shape[0]):
    #     for y in range(img_y.shape[0]):
    #         # only replace pixels that are actually visible in the image
    #         # this means we only replace pixels starting from the upper left corner (0,0)
    #         # until the lower right corner (height, width) of the original image
    #         # any pixels outside of the original image are ignored
    #         if img_x[x][y] >= 0. and img_y[x][y] >= 0.: 
    #             if img_x[x][y] < image_size[2] and img_y[x][y] < image_size[3]:
    #                 # print(x, y)
    #                 # print(img_x[x,y], img_y[x,y])
    #                 bit_mask[int(img_x[x][y])][int(img_y[x][y])] = 0.


    #img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)
    # get ones where there's a patch in the image
    #ones = torch.ones(patch_size)
    #bit_mask = grid_sample(ones, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    # invert the bit mask, such that there's zeros where there's a patch and ones where it isn't
    #bit_mask = 1 - bit_mask

    #plt.imshow(bit_mask.detach().numpy())
    return bit_mask

def get_transformed_patch(grid, patch, image_size):
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
    transformed_patch = torch.zeros((image_size[2:]))

    #transformed_patch = grid_sample(patch, grid, mode="nearest", padding_mode="zeros", align_corners=False)


    # for ind_x, x in enumerate(img_x):
    #     for ind_y, y in enumerate(img_y):
    #         if ind_x
    #         image[][] = patch[ind_x][ind_y]
    # print("--beginning of transformed patch---")
    # print(grid.shape)
    # print(grid.reshape(2, 100, 100).shape)
    img_x, img_y = grid.reshape(2, 100, 100)


    for x in range(img_x.shape[0]):
        for y in range(img_y.shape[1]):
            # only replace pixels that are actually visible in the image
            # this means we only replace pixels starting from the upper left corner (0,0)
            # until the lower right corner (height, width) of the original image
            # any pixels outside img_x, img_y
            transformed_patch[img_x[x][y]][img_y[x][y]] = patch[x][y]

    return transformed_patch
    

def place_patch(image, patch, angle, scale, tx, ty):
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

    mask = torch.ones_like(patch)

    # T_patch_in_attacker = calc_T_in_attaker_frame(patch_size=patch_size[2:])
    
    # T_attacker_in_camera = calc_T_attacker_in_camera(attacker_pose[:3], attacker_pose[3:])

    # coords_grid = affine_grid(patch_size=patch_size, image_size=image_size, camera_config=camera_config, 
    #                T_attacker_in_camera=T_attacker_in_camera, T_patch_in_attacker=T_patch_in_attacker)

    # create a 3x3 rotation and translation matrix
    rotation_matrix = torch.zeros(1, 3, 3)
    
    cos = torch.cos(angle) # angle in radians
    sin = torch.sin(angle)

    rotation_matrix[:, 0, 0] = cos
    rotation_matrix[:, 0, 1] = -sin
    rotation_matrix[:, 1, 0] = sin
    rotation_matrix[:, 1, 1] = cos
    rotation_matrix[:, 0, 2] = tx
    rotation_matrix[:, 1, 2] = ty
    rotation_matrix[:, 2, 2] = 1

    # create a 3x3 scaling matrix
    scaling = torch.zeros(1, 3, 3)

    scaling[:, 0, 0] = scale
    scaling[:, 1, 1] = scale
    scaling[:, 2, 2] = 1

    # calculate full transformation matrix
    transformation_matrix = rotation_matrix @ scaling

    # PyTorch's affine grid funtion needs the inverse of the 3x3 transformation matrix
    inv_t_matrix = torch.inverse(transformation_matrix)[:, :2] # affine grid expects only the first 2 rows, the last row (0, 0, 1) is neglected
    affine_grid = torch.nn.functional.affine_grid(inv_t_matrix, size=(1, 1, 96, 160), align_corners=False)

    # calculate both the bit mask and the transformed patch
    bit_mask = grid_sample(mask, affine_grid).bool()
    transformed_patch = grid_sample(patch, affine_grid)

    # first erase all pixel values in the original image in the area of the patch
    modified_image = image * ~bit_mask
    # and now replace these values with the transformed patch
    modified_image += transformed_patch

    return modified_image

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
    # we'll extend the shape by one dimension to work with batches of images -> there's only one image so batch_no == 1
    image = image.unsqueeze(0)
    # the pixel values of the images range from 0 to 255
    # pose is stored as x, y, z, phi -> phi is not needed for patch placement
    pose = pose[:3]

    # load the camera parameters from the config yaml file
    # import yaml
    # with open('adversarial_frontnet/camera_calibration/camera_config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # and convert to numpy arrays
    # camera_matrix = torch.tensor(config['camera_matrix'])
    # translation_marix = torch.tensor(config['translation_matrix'])
    # dist_coeffs = torch.tensor(config['dist_coeffs'])

    # generate a random patch
    # first, generate random values between 0 and 1, 
    # then multiply by 255. to receive values between 0. and 255.
    patch = (torch.rand(1, 1, 50, 50) * 255.).requires_grad_()


    # set an arbitrary pose for testing
    # pose includes x,y,z and quaternions in order qx, qy, qz, qw
    # x,y,z will be the center of the patch in camera frame
    # quaternions hold rotation information from attacker (holding the patch) to camera frame
    # pose = torch.tensor([2.7143893241882324,1.6456797122955322,0.4578791558742523, 
    #                      0.0114797880217894, 0.0744068142306854, -0.1520472288581698, 0.985501639095322]).requires_grad_()

    # calculate random translation, rotation (in radians), and scale factor
    tx = torch.randint(high=2, size=(1,)).float().requires_grad_()
    ty = torch.randint(high=2, size=(1,)).float().requires_grad_()
    rotation = torch.distributions.uniform.Uniform(np.radians(-45), np.radians(45)).sample().requires_grad_()  # PyTorch doesn't offer a radians function yet
    scale = torch.distributions.uniform.Uniform(0.01, 1).sample().requires_grad_()
    
    # place the patch
    new_image = place_patch(image, patch, angle=rotation, scale=scale, tx=tx, ty=ty)
    print(new_image.shape)
    # check if image still keeps the gradient
    print(new_image.requires_grad)
    # plot the final image
    plt.imshow(new_image[0][0].detach().numpy())
    plt.show()
