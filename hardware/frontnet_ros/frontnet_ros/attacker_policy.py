import numpy as np
import rowan

import matplotlib.pyplot as plt

# black_p = np.zeros((1, 96, 160))
# white_p = np.ones((1, 96, 160))
# patches = np.stack([black_p, white_p])

def gen_circle(r, n):
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.array([x, y]).T


# all target poses 
target_positions = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 0], [2, 0, 0]])
num_targets = len(target_positions)

# example opimized position norm
# positions_p0 = np.array([[0.5, -0.6, 0.4], [0.3, -0.2, -1.0]])
# positions_p1 = np.array([[0.5, -0.8, 0.3], [0.5, 0.6, 0.4]])
# positions = np.stack([positions_p0, positions_p1])

# current victim pose
victim_pose = np.array([[-1., 0., 0., 0.],        # initial pose in world frame 
                          [0., -1., 0., 0.],        # roll = 0, pitch = 0, yaw=0 
                          [0., 0., 1., 1.]])         # no rotation, z = 1


# desired next victim pose in world frame
desired_trajectory = gen_circle(2., 20)
desired_pose = np.array([[-1., 0., 0., 0.],          # desired pose in world frame 
                            [0., -1., 0., 0.],         # 
                            [0., 0., 1., 1.]])         # no rotation, z = 1

desired_idx = np.random.randint(low= 0, high=len(desired_trajectory)-1)
desired_pose[:2, 3] = desired_trajectory[desired_idx]

# assignment
A = np.array([[True, True, False, False], [False, False, True, True]])

all_victim_positions = []
all_desired_positions = []

all_victim_positions.append(victim_pose[:2, 3].T)
all_desired_positions.append(desired_pose[:2, 3].T)

for i in range(300):
    print(f"------{i}------")              
    victim_pos= victim_pose[:2, 3]

    victim_quat = rowan.from_matrix(victim_pose[:3, :3])


    # desired_yaw should be 0
    #r, p, desired_yaw = rowan.to_euler(rowan.from_matrix(victim_pose[:3,:3] @ desired_pose[:3, :3]))
    desired_yaw = 0.

    # frontnet_out = np.array([1, -1, 0]) + np.random.rand(3,)
    # print("Frontnet prediction: ", frontnet_out)


    possible_new_positions = np.zeros((num_targets, 3, 4))
    # print(possible_new_positions)

    for target_idx in range(num_targets):
        # since we're keeping the yaw angle of the victim fixed, 
        # we can simply compute the max. achievable new position in world coordinates:
        max_new_pos = victim_pos + target_positions[target_idx][:2]
        max_new_pos[0] -= 1. # safety radius
        max_new_pos[0] -= np.random.uniform(0., 0.2) # debug

        new_pose_world = np.zeros((3, 4))
        new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
        new_pose_world[:2, 3] = max_new_pos

        possible_new_positions[target_idx, :, :] = new_pose_world


    # maybe not even needed
    #     print("target: ", target_positions[target_idx])
    #     print("victim_pos: ", victim_pos)
    #     new_pos = rowan.rotate(victim_quat, target_positions[target_idx])
    #     print("new_pos:", new_pos)
    #     heading_vec = [1*np.cos(desired_yaw), 1*np.sin(desired_yaw), 0.]
    #     print("heading_vec:", heading_vec)

    #     new_pos_world = new_pos + victim_pos + heading_vec
    #     print("new_pos in world: ", new_pos_world)
    #     new_pose_world = np.zeros((3, 4))
    #     new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
    #     new_pose_world[..., 3][:3] = new_pos_world 
    #     # print("new pose in world: ", new_pose_world)
    #     possible_new_positions[target_idx, :, :] = new_pose_world

    # # print(possible_new_positions)

    distances = [np.linalg.norm(possible_new_positions[target_idx]-desired_pose, ord=2) for target_idx in range(num_targets)]
    print("l2 dist to desired: ", distances)
    k = np.argmin(distances)


    print("k: ", k)
    # print("Assignment: ", A[..., k])
    m = np.argmax(A[..., k])
    print("m: ", m)
    # print(f"Presenting patch {m} at position {positions[m, k]} (in image frame)")

    # pretend that pose got updated by frontnet output
    # the pose will later be read from the mocap system
    position_change = possible_new_positions[k] - victim_pose
    victim_pose[:2, 3] += (0.3*(position_change[:2, 3]) + np.random.uniform(-0.3, 0.3, (2,))) # add a bit of uncertainty
    print("updated victim pose: ", victim_pose)
    dist_after_update = np.linalg.norm((victim_pose-desired_pose), ord=2)
    print("error to desired: ", dist_after_update)
    

    # distances = [np.linalg.norm(victim_pos-desired_trajectory[d_idx]) for d_idx in range(len(desired_trajectory))]
    
    # # desired_idx = np.argmin(distances)
    # #desired_pose[:2, 3] = desired_trajectory[desired_idx]

    if dist_after_update <= 0.2:
        if desired_idx < len(desired_trajectory)-1:
            desired_idx += 1
        else:
            desired_idx = 0
        desired_pose[:2, 3] = desired_trajectory[desired_idx]
        print("updated desired pose! ", desired_pose)

    all_victim_positions.append(victim_pose[:2, 3].copy())
    all_desired_positions.append(desired_pose[:2, 3].copy())


all_victim_positions = np.array(all_victim_positions)
all_desired_positions = np.array(all_desired_positions)



fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-5., 5)
ax.plot(desired_trajectory.T[1], desired_trajectory.T[0])
ax.plot(all_victim_positions.T[1],  all_victim_positions.T[0], label='victim position')#, marker=(3, 0, desired_yaw), markersize=20, linestyle='None')
# for i, (a,b) in enumerate(zip(all_victim_positions.T[1], all_victim_positions.T[0])): 
#     plt.text(a, b-0.007, str(i))

ax.plot(all_desired_positions.T[1], all_desired_positions.T[0], marker=(3, 0, desired_yaw), markersize=20, linestyle='None', label='desired checkpoints')
ax.legend()
plt.show()