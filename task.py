import numpy as np
from physics_sim import PhysicsSim
import random

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, ):
        """Initialize a Task object.
        Params
        ======
        """
        # Simulation
        self.runtime = 5.
        self.init()
        self.action_repeat = 1

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        # self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        dist = np.linalg.norm(np.array(self.sim.pose[:3])-np.array(self.target_pos))
        reward = 1.-.2*(dist).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(np.concatenate([self.sim.pose, self.target_pos]))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def init(self):
        self.init_pose = np.array(np.concatenate(
          [[float(random.randint(0,10)) for x in range(3)], [0., 0., 0.]]))
        self.target_pos = np.array([ float(random.randint(0,10)) for x in range(3)])
        self.sim = PhysicsSim(self.init_pose, None, None, self.runtime)

    def reset(self):
        """Reset the sim to start a new episode."""
        self.init()
        self.sim.reset()
        state = np.concatenate([np.concatenate([self.sim.pose, self.target_pos])] * self.action_repeat)
        return state
