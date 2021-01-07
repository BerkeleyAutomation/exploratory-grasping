"""
The environment for grasp exploration. Defines states, actions, rewards
and steps forward based on the reward and toppling transitions.
"""

import numpy as np

class StablePoseEnv(object):

    def __init__(self,
                 arm_means,
                 pose_probs,
                 topple_matrix):

        self.arm_means = arm_means
        self.num_poses, self.num_arms = self.arm_means.shape
        self.pose_probs = pose_probs / pose_probs.sum()
        self.topple_matrix = topple_matrix.cumsum(axis=1)

        # Draw pose from pose distribution
        self.start_pose = np.random.choice(np.arange(self.num_poses),
                                           p=self.pose_probs)
        self.pose = self.start_pose

    def step(self, arm):

        if arm < self.num_arms:
            reward = np.random.random() < self.arm_means[self.pose, arm]
        else:
            raise IndexError("Arm idx out of bounds")

        # Update pose if reward is 1 or we topple
        if reward:
            self.pose = np.random.choice(np.arange(self.num_poses),
                                         p=self.pose_probs)
        else:
            self.pose = (np.random.random() < self.topple_matrix[self.pose]).argmax()
        
        return reward

    def reset(self, start_pose=None):
        if start_pose is not None:
            self.pose = self.start_pose
        else:
            self.pose = np.random.choice(np.arange(self.num_poses),
                                         p=self.pose_probs)
