#!/usr/bin/env python

import numpy as np
from pathlib import Path
from geometry_msgs.msg import PoseStamped
from functools import partial


from crazyflie_py import *
from crazyflie_py.uav_trajectory import Trajectory

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

        
    def pose_callback(self, store_into, msg: PoseStamped):
        # print(msg)
        # store the latest pose
        if store_into == 'a':
            self.pose_a = msg
        else:
            self.pose_v = msg

    def compute_attacker_pose(self, pos_v_desired):
        # can use
        # self.pose_a: current pose of attacker
        # self.pose_v: current pose of victim
        # print(self.pose_a)
        # print(self.pose_v)

        # return desired pos and yaw for the attacker
        return pos_v_desired, 0

    def run(self):
        offset=np.zeros(3)
        rate=10
        stretch = 10 # >1 -> slower

        traj = Trajectory()
        traj.loadcsv("/home/pia/Documents/Coding/adversarial_frontnet/hardware/frontnet_ros/data/circle0.csv")#Path(__file__).parent / "data/circle0.csv")

        start_time = self.timeHelper.time()
        while not self.timeHelper.isShutdown():
            t = self.timeHelper.time() - start_time
            if t > traj.duration * stretch:
                break

            e = traj.eval(t / stretch)

            pos_v_desired = e.pos + offset

            pos_a_desired, yaw_a_desired = self.compute_attacker_pose(pos_v_desired)
            self.cf_a.cmdFullState(
                pos_a_desired,
                np.zeros(3),
                np.zeros(3),
                yaw_a_desired,
                np.zeros(3))

            self.timeHelper.sleepForRate(rate)

        self.cf_a.notifySetpointsStop()
        self.cf_a.land(targetHeight=0.03, duration=3.0)
        self.timeHelper.sleep(3.0)


def main():
    a = Attack()
    a.run()


if __name__ == "__main__":
    main()
