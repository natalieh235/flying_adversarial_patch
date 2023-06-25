#!/usr/bin/env python

import numpy as np
import rowan
from pathlib import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from functools import partial


from crazyflie_py import *
from crazyflie_py.uav_trajectory import Trajectory

from tf2_ros import TransformBroadcaster

import yaml


import sys
sys.path.insert(0,'/home/pia/Documents/Coding/adversarial_frontnet/hardware/frontnet_ros/frontnet_ros/')
from attacker_policy import gen_transformation_matrix, get_bb_patch, xyz_from_bb


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



def calc_max_possible_victim_change(current_victim_pose, target_position):
    max_new_pos = current_victim_pose[:2, 3] + target_position[:2]   # v_x + target_x, v_y + target_y
    max_new_pos[0] -= 1. # substract safety radius

    victim_quat = rowan.from_matrix(current_victim_pose[:3, :3])

    new_pose_world = np.zeros((4, 4))
    new_pose_world[:3, :3] = rowan.to_matrix(victim_quat)
    new_pose_world[:2, 3] = max_new_pos
    new_pose_world[-1, -1] = 1.

    return new_pose_world

def update_attacker_pose(T_victim_world_c, T_victim_world_d):

    target_positions = np.array([[1, 1, 0], [1, -1, 0]])
    
    A = np.array([[True, False], [False, True]])
    # possible_attacker_position = np.array([[0.28316405,  0.20245424, -0.2117372 ],
    #                                        [0.30532456, -0.20091306, -0.22638479]])



    T_attacker_in_world_d = {'matrices': [], 'distances': []}
    for k in range(A.shape[1]):

        m = np.argwhere(A[..., k]==True)[0, 0]

        # transformation_matrix = gen_transformation_matrix(*optimized_patch_positions[m,k])

        # bounding_box_placed_patch = get_bb_patch(transformation_matrix)

        # patch_in_camera = xyz_from_bb(bounding_box_placed_patch, camera_intrinsics)
        # # print(patch_in_camera)
        # patch_in_victim = (np.linalg.inv(camera_extrinsics) @ [*patch_in_camera, 1])[:3]
        # print(patch_in_victim)
        # print(patch_rel_position)T_victim_world_c

        # T_patch_in_victim = np.zeros((4,4))
        # T_patch_in_victim[:3, :3] = np.eye(3,3)
        # T_patch_in_victim[:3, 3] = patch_in_victim
        # T_patch_in_victim[-1, -1] = 1.

        # print(T_patch_in_victim)

        #T_patch_in_world = T_victim_world @ T_patch_in_victim   <--- seems faulty to me
        T_patch_in_world = T_victim_world_c.copy()
        T_patch_in_world[:3, 3] += possible_attacker_position[k]
        # print(T_patch_in_world)

        T_attacker_in_world = T_patch_in_world.copy()
        T_attacker_in_world[2, 3] += 0.096  # center of CF is 9.6 cm above center of patch

        T_attacker_in_world_d['matrices'].append(T_attacker_in_world)

        T_possible_new_victim_pose = calc_max_possible_victim_change(T_victim_world_c, target_positions[k])

        # print(T_victim_world_c.shape, T_victim_world_d.shape, T_possible_new_victim_pose.shape)

        # Why base this decision on the victim pose?
        # "The objective is to minimize the tracking error of the victim
        # between its current pose pv and desired pose  ̄pv , i.e.∫t ∥ ̄pv (t) − ˆpv (t)∥dt."
        T_attacker_in_world_d['distances'].append(np.linalg.norm((T_victim_world_d-T_possible_new_victim_pose), ord=2))

    return T_attacker_in_world_d['matrices'][np.argmin(T_attacker_in_world_d['distances'])]

class Attack():
    attacker_id = 4
    victim_id = 231

    def __init__(self):
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.node = self.swarm.allcfs

        self.pose_a = None
        self.subscription_a = self.node.create_subscription(
                PoseStamped,
                f'cf{self.attacker_id}/pose',
                partial(self.pose_callback, "a"),
                10)
        
        self.pose_v = None
        self.subscription_v = self.node.create_subscription(
                PoseStamped,
                f'cf{self.victim_id}/pose',
                partial(self.pose_callback, "v"),
                10)
        
        self.cf_a = self.swarm.allcfs.crazyfliesById[self.attacker_id]
        self.cf_v = self.swarm.allcfs.crazyfliesById[self.victim_id]

        self.tfbr = TransformBroadcaster(self.node)

        
    def pose_callback(self, store_into, msg: PoseStamped):
        # print(msg)
        # store the latest pose
        if store_into == 'a':
            self.pose_a = msg
        else:
            self.pose_v = msg

    def _broadcast(self, name, frame, T):
        t_base = TransformStamped()
        t_base.header.stamp = self.node.get_clock().now().to_msg()
        t_base.header.frame_id = frame
        t_base.child_frame_id = name

        qw, qx, qy, qz = rowan.from_matrix(T[:3, :3])
        x, y, z = T[:3, 3]


        t_base.transform.translation.x = x
        t_base.transform.translation.y = y
        t_base.transform.translation.z = z
        t_base.transform.rotation.x = qx
        t_base.transform.rotation.y = qy
        t_base.transform.rotation.z = qz
        t_base.transform.rotation.w = qw
        self.tfbr.sendTransform(t_base)

    def _get_T(self, msg):
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]

        rot = rowan.to_matrix(quat)
        T = np.zeros((4,4))
        T[:3, :3] = rot
        T[:3, 3] = pos
        T[-1, -1] = 1
        return T

    def compute_attacker_pose(self, pos_v_desired, rel_attacker_pos):
        # can use
        # self.pose_a: current pose of attacker
        # self.pose_v: current pose of victim
        # print(self.pose_a)
        # print(self.pose_v)

        
        # pos_v = [self.pose_v.position.x, self.pose_v.position.y, self.pose_v.position.z]
        # quat_v = [self.pose_v.orientation.]


        T_victim_world_c = self._get_T(self.pose_v)

        self._broadcast('v', 'world', T_victim_world_c)

        T_attacker_world = self._get_T(self.pose_a)

        self._broadcast('a', 'world', T_attacker_world)

        # possible_patch_position = np.array([[0.28316405,  0.20245424, -0.2117372 ],
        #                                     [0.30532456, -0.20091306, -0.22638479]])

        T_patch_in_victim = np.eye(4)
        T_patch_in_victim[:3, 3] = rel_attacker_pos
        T_patch_in_victim[2, 3] += 0.096  # debug

        # print("victim in world: ", T_victim_world_c)
        # print("pacth in victim: ", T_patch_in_victim)

        T_victim_world_d = np.eye(4)
        T_victim_world_d[:3, 3] = pos_v_desired

        self._broadcast('vd', 'world', T_victim_world_d)

        T_attacker_in_world = T_victim_world_d @ T_patch_in_victim
        # print("attacker in world: ", T_attacker_in_world)

        self._broadcast('ad', 'world', T_attacker_in_world)
        # self._broadcast('ad2', 'v', T_patch_in_victim)


        # get rotations from quaternions
        # build whole 4x4 pose matrix
        # update pose

        # T_attacker_world = update_attacker_pose

        # return desired pos and yaw for the attacker
        roll, pitch, yaw = rowan.to_euler(rowan.from_matrix(T_attacker_in_world[:3, :3]), 'xyz')
        return T_attacker_in_world[:3, 3], yaw

    def run(self, rel_attacker_pos):
        offset=np.zeros(3)
        rate=1
        stretch = 1000 # >1 -> slower

        traj = Trajectory()
        traj.loadcsv("/home/pia/Documents/Coding/adversarial_frontnet/hardware/frontnet_ros/data/circle0.csv")#Path(__file__).parent / "data/circle0.csv")

        while True:
            if self.pose_a is not None and self.pose_v is not None:
                break
            self.timeHelper.sleep(0.1)

        start_time = self.timeHelper.time()
        while not self.timeHelper.isShutdown():
            t = self.timeHelper.time() - start_time
            if t > traj.duration * stretch:
                break

            # e = traj.eval(t / stretch)
            # pos_v_desired = e.pos + offset
            pos_v_desired = np.array([0., 0., 1.], dtype=np.float32)

            pos_a_desired, yaw_a_desired = self.compute_attacker_pose(pos_v_desired, rel_attacker_pos)
            
            self.cf_a.goTo(pos_a_desired, yaw_a_desired, 2.)
            # self.cf_a.cmdFullState(
            #     pos_a_desired,
            #     np.zeros(3),
            #     np.zeros(3),
            #     yaw_a_desired,
            #     np.zeros(3))

            self.timeHelper.sleepForRate(rate)

        self.cf_a.notifySetpointsStop()
        self.cf_a.land(targetHeight=0.03, duration=3.0)
        self.timeHelper.sleep(3.0)


def main():
    a = Attack()

    # positions_p0 = np.array([[0.5, -0.6, 0.4], [0.3, -0.2, -1.0]])
    # positions_p1 = np.array([[0.5, -0.8, 0.3], [0.5, 0.6, 0.4]])
    # patch_positions_image = np.stack([positions_p0, positions_p1])
    position_p0 = np.array([0.5, 0.5, 0.5])

 
    with open('/home/pia/Documents/Coding/adversarial_frontnet/misc/camera_calibration/calibration.yaml') as f:
        camera_config = yaml.load(f, Loader=yaml.FullLoader)

    camera_intrinsic = np.array(camera_config['camera_matrix'])
    distortion_coeffs = np.array(camera_config['dist_coeff'])
    
    rvec = np.array(camera_config['rvec'])
    tvec = camera_config['tvec']

    camera_extrinsic = np.zeros((4,4))
    camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
    camera_extrinsic[:3, 3] = tvec
    camera_extrinsic[-1, -1] = 1.

    print(camera_intrinsic)
    print(camera_extrinsic)
    
    transformation_matrix = gen_transformation_matrix(*position_p0)

    bounding_box_placed_patch = get_bb_patch(transformation_matrix)

    patch_in_camera = xyz_from_bb(bounding_box_placed_patch, camera_intrinsic, distortion_coeffs)
    # print(patch_in_camera)
    patch_in_victim = (np.linalg.inv(camera_extrinsic) @ [*patch_in_camera, 1])[:3]

    print(patch_in_victim)


    a.run(patch_in_victim)


if __name__ == "__main__":
    main()
