import numpy as np
import rowan

# black_p = np.zeros((1, 96, 160))
# white_p = np.ones((1, 96, 160))
# patches = np.stack([black_p, white_p])


# all target poses 
target_poses = np.array([[1, 1, 0], [1, -1, 0]])
num_targets = len(target_poses)

# example opimized position norm
positions_p0 = np.array([[0.5, -0.6, 0.4], [0.3, -0.2, -1.0]])
positions_p1 = np.array([[0.5, -0.8, 0.3], [0.5, 0.6, 0.4]])
positions = np.stack([positions_p0, positions_p1])

# current victim pose
victim_pose = np.array([[-1., 0., 0., 0.],        # initial pose in world frame 
                          [0., -1., 0., 0.],        # roll = 0, pitch = 0, yaw=0 
                          [0., 0., 1., 1.]])         # no rotation, z = 1


# desired next victim pose in world frame
desired_pose = np.array([[-1., 0., 0., 0.],         # desired pose in world frame 
                            [0., -1., 0., 1.],         # y = 1
                            [0., 0., 1., 1.]])         # no rotation, z = 1

# assignment
A = np.array([[True, False], [False, True]])


for i in range(10):                          
    victim_pos= victim_pose[..., 3]

    victim_quat = rowan.from_matrix(victim_pose[:3, :3])


    # desired_yaw should be 0
    r, p, desired_yaw = rowan.to_euler(rowan.from_matrix(victim_pose[:3,:3] @ desired_pose[:3, :3]))

    # frontnet_out = np.array([1, -1, 0]) + np.random.rand(3,)
    # print("Frontnet prediction: ", frontnet_out)


    possible_new_positions = np.zeros((num_targets, 3, 4))
    # print(possible_new_positions)

    for target_idx in range(num_targets):
        new_pos = rowan.rotate(victim_quat, target_poses[target_idx])
        heading_vec = [1*np.cos(desired_yaw), 1*np.sin(desired_yaw), 0.]

        new_pos_world = new_pos + victim_pos + heading_vec
        new_pose_world = np.zeros((3, 4))
        new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
        new_pose_world[..., 3][:3] = new_pos_world 
        possible_new_positions[target_idx, :, :] = new_pose_world

    # print(possible_new_positions)

    distances = [np.linalg.norm(possible_new_positions[target_idx]-desired_pose, ord=2) for target_idx in range(num_targets)]
    better_target_idx = np.argmin(distances)

    print("target idx: ", better_target_idx)
    better_patch_idx = np.argmax(A[..., better_target_idx])
    print(f"Presenting patch {better_patch_idx} at position {positions[better_patch_idx, better_target_idx]} (in image frame)")

    victim_pose[:2, 3] += 0.2*possible_new_positions[better_target_idx][:2, 3]
    print("updated victim pose: ", victim_pose)
    print("error to desired: ", np.linalg.norm((victim_pose-desired_pose), ord=2))