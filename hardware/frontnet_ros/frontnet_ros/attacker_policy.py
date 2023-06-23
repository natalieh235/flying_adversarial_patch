import numpy as np
import rowan

import matplotlib.pyplot as plt
import cv2

import yaml

import torch

black_p = np.zeros((1, 96, 160))
white_p = np.ones((1, 96, 160))
rand_p = np.random.rand(1, 96, 160)
# patches = np.stack([black_p, white_p])

def gen_circle(r, n):
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.array([x, y]).T

def gen_transformation_matrix(sf, tx, ty):
    matrix = np.zeros((3,3))
    matrix[0,0] = sf
    matrix[1,1] = sf
    matrix[0, 2] = tx
    matrix[1, 2] = ty
    matrix[2, 2] = 1.
    return matrix

def get_bb_patch(transformation):
    transformation_t = torch.tensor(transformation).unsqueeze(0)
    inv_transform = torch.inverse(transformation_t)[:, :2]
    affine_grid = torch.nn.functional.affine_grid(inv_transform, size=(1, 1, 96, 160), align_corners=True)

    transformed_patch = torch.nn.functional.grid_sample(torch.ones(1, 1, 96, 160).double(), affine_grid, align_corners=True, padding_mode='zeros')
    

    # corner_left = [(corner_left[1] +1 * 0.5 * 160).round().item(), (corner_left[0] +1 * 0.5 * 96).round().item()]
    # corner_right = [(corner_right[1] +1 * 0.5 * 160).round().item(), (corner_right[0] +1 * 0.5 * 96).round().item()] 
    # print(corner_left, corner_right)

    patch_coords = torch.nonzero(transformed_patch)[..., 2:]
    print(patch_coords, patch_coords.shape)
    # patch_h = (patch_coords[-1][0]-patch_coords[0][0]).round().item()
    # patch_w = (patch_coords[-1][1].item()-patch_coords[0][1]).round().item()
    # corner_left = [patch_coords[0][1].item(), patch_coords[0][0].item() + int(patch_h/2)]
    # corner_right = [patch_coords[-1][1].item(), patch_coords[0][0].item() + int(patch_h/2)]
    # corner_left = [patch_coords[0][1].item(), (patch_coords[-1][0]-patch_coords[0][0]).round().item()]
    # corner_right = [, (patch_coords[0][0]+(patch_coords[-1][0]-patch_coords[0][0])).round().item()]
    #return np.array([corner_left, corner_right])
    xmin = patch_coords[0][1].item()
    ymin = patch_coords[0][0].item()
    xmax = patch_coords[-1][1].item()
    ymax = patch_coords[-1][0].item()

    return [xmin, ymin, xmax, ymax]


# printed patch width = 28.2 cm
RADIUS = 0.14 # in m

# compute 
def xyz_from_bb(bb,mtrx,dist_vec):
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
    P1_rec = cv2.undistortPoints(P1, mtrx, dist_vec, None, mtrx).flatten()
    P2_rec = cv2.undistortPoints(P2, mtrx, dist_vec, None, mtrx).flatten()

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


# camera_extrinsics = np.load("/home/pia/Documents/Coding/adversarial_frontnet/misc/aideck7/extrinsic.npy")
with open('misc/aideck7/camera_config.yaml') as f:
    camera_config = yaml.load(f, Loader=yaml.FullLoader)

camera_matrix = np.array(camera_config['camera_matrix'])
dist_vec = np.array(camera_config['dist_coeff'])


# all target poses 
target_positions = np.array([[1, 1, 0], [1, -1, 0]])#, [1, 0, 0], [2, 0, 0]])
num_targets = len(target_positions)

# example opimized position norm
positions_p0 = np.array([[0.5, -0.6, 0.4], [0.3, -0.2, -1.0]])
positions_p1 = np.array([[0.5, -0.8, 0.3], [0.5, 0.6, 0.4]])
positions = np.stack([positions_p0, positions_p1])

transformation_matrix = gen_transformation_matrix(*positions_p0[0])

bounding_box_placed_patch = get_bb_patch(transformation_matrix)

print(bounding_box_placed_patch)

patch_rel_position = xyz_from_bb(bounding_box_placed_patch, camera_matrix, dist_vec)
print(patch_rel_position)



# current victim pose
victim_pose = np.array([[-1., 0., 0., 0.],        # initial pose in world frame 
                          [0., -1., 0., 0.],        # roll = 0, pitch = 0, yaw=0 
                          [0., 0., 1., 1.]])         # no rotation, z = 1


# desired next victim pose in world frame
#desired_trajectory = gen_circle(2., 20)
desired_pose = np.array([[-1., 0., 0., 0.],          # desired pose in world frame 
                            [0., -1., 0., 0.],         # 
                            [0., 0., 1., 1.]])         # no rotation, z = 1

#desired_idx = np.random.randint(low= 0, high=len(desired_trajectory)-1)
#desired_pose[:2, 3] = desired_trajectory[desired_idx]

# assignment
#A = np.array([[True, True, False, False], [False, False, True, True]])
A = np.array([[True, False], [False, True]])

all_victim_positions = []
all_desired_positions = []

all_victim_positions.append(victim_pose[:2, 3].T)
all_desired_positions.append(desired_pose[:2, 3].T)

# for i in range(5):
#     for target_idx in range(num_targets):



#     print(f"------{i}------")              
#     victim_pos= victim_pose[:2, 3]

#     victim_quat = rowan.from_matrix(victim_pose[:3, :3])


#     # desired_yaw should be 0
#     #r, p, desired_yaw = rowan.to_euler(rowan.from_matrix(victim_pose[:3,:3] @ desired_pose[:3, :3]))
#     desired_yaw = 0.

#     # frontnet_out = np.array([1, -1, 0]) + np.random.rand(3,)
#     # print("Frontnet prediction: ", frontnet_out)


#     possible_new_positions = np.zeros((num_targets, 3, 4))
#     # print(possible_new_positions)

#     for target_idx in range(num_targets):
#         # since we're keeping the yaw angle of the victim fixed, 
#         # we can simply compute the max. achievable new position in world coordinates:
#         max_new_pos = victim_pos + target_positions[target_idx][:2]
#         max_new_pos[0] -= 1. # safety radius
#         max_new_pos[0] -= np.random.uniform(0., 0.2) # debug

#         new_pose_world = np.zeros((3, 4))
#         new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
#         new_pose_world[:2, 3] = max_new_pos

#         possible_new_positions[target_idx, :, :] = new_pose_world


#     # maybe not even needed
#     #     print("target: ", target_positions[target_idx])
#     #     print("victim_pos: ", victim_pos)
#     #     new_pos = rowan.rotate(victim_quat, target_positions[target_idx])
#     #     print("new_pos:", new_pos)
#     #     heading_vec = [1*np.cos(desired_yaw), 1*np.sin(desired_yaw), 0.]
#     #     print("heading_vec:", heading_vec)

#     #     new_pos_world = new_pos + victim_pos + heading_vec
#     #     print("new_pos in world: ", new_pos_world)
#     #     new_pose_world = np.zeros((3, 4))
#     #     new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
#     #     new_pose_world[..., 3][:3] = new_pos_world 
#     #     # print("new pose in world: ", new_pose_world)
#     #     possible_new_positions[target_idx, :, :] = new_pose_world

#     # # print(possible_new_positions)

#     distances = [np.linalg.norm(possible_new_positions[target_idx]-desired_pose, ord=2) for target_idx in range(num_targets)]
#     print("l2 dist to desired: ", distances)
#     k = np.argmin(distances)


#     print("k: ", k)
#     # print("Assignment: ", A[..., k])
#     m = np.argmax(A[..., k])
#     print("m: ", m)
#     # print(f"Presenting patch {m} at position {positions[m, k]} (in image frame)")

#     # pretend that pose got updated by frontnet output
#     # the pose will later be read from the mocap system
#     position_change = possible_new_positions[k] - victim_pose
#     victim_pose[:2, 3] += (0.3*(position_change[:2, 3]) + np.random.uniform(-0.3, 0.3, (2,))) # add a bit of uncertainty
#     print("updated victim pose: ", victim_pose)
#     dist_after_update = np.linalg.norm((victim_pose-desired_pose), ord=2)
#     print("error to desired: ", dist_after_update)
    

#     # distances = [np.linalg.norm(victim_pos-desired_trajectory[d_idx]) for d_idx in range(len(desired_trajectory))]
    
#     # # desired_idx = np.argmin(distances)
#     # #desired_pose[:2, 3] = desired_trajectory[desired_idx]

#     if dist_after_update <= 0.2:
#         if desired_idx < len(desired_trajectory)-1:
#             desired_idx += 1
#         else:
#             desired_idx = 0
#         desired_pose[:2, 3] = desired_trajectory[desired_idx]
#         print("updated desired pose! ", desired_pose)

#     all_victim_positions.append(victim_pose[:2, 3].copy())
#     all_desired_positions.append(desired_pose[:2, 3].copy())


# all_victim_positions = np.array(all_victim_positions)
# all_desired_positions = np.array(all_desired_positions)



# fig, ax = plt.subplots()
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5., 5)
# ax.plot(desired_trajectory.T[1], desired_trajectory.T[0])
# ax.plot(all_victim_positions.T[1],  all_victim_positions.T[0], label='victim position')#, marker=(3, 0, desired_yaw), markersize=20, linestyle='None')
# # for i, (a,b) in enumerate(zip(all_victim_positions.T[1], all_victim_positions.T[0])): 
# #     plt.text(a, b-0.007, str(i))

# ax.plot(all_desired_positions.T[1], all_desired_positions.T[0], marker=(3, 0, desired_yaw), markersize=20, linestyle='None', label='desired checkpoints')
# ax.legend()
# plt.show()