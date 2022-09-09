from pprint import pprint

import imageio as imageio
import time

from highway_env import utils

import gym
import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete
from highway_env.envs.common.observation import KinematicObservation
from rewards import reward_register
from rewards.reward_wrapper import HighwayRewardWrapper

class KinematicsExtension(gym.Wrapper):

    ACCELERATION_RANGE = [-5.0, 5.0]

    def __init__(self, env, number_of_vehicles, number_of_lanes):
        super().__init__(env)
        self.number_of_vehicles = number_of_vehicles
        self.number_of_lanes = number_of_lanes
        self.acceleration_range = self.ACCELERATION_RANGE
        self.previous_action = [0, 0]
        self.previous_observation = env.reset()
        self.dt = 1 / env.config["simulation_frequency"]
        self.modifies_observation_log = dict()
        self.time_step = 0

        '''
        self.observation_space = gym.spaces.Dict(dict(
            vehicle_id=gym.spaces.Dict(dict(
            x=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            y=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            v_lat=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            v_lon=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            a_lat=Box(low=self.acceleration_range[0], high=self.acceleration_range[1], shape=(1,), dtype=np.float32),
            a_lon=Box(low=self.acceleration_range[0], high=self.acceleration_range[1], shape=(1,), dtype=np.float32),
            j_lat=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            j_lon=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32)
        ))))
        '''

    def get_modified_obs(self, obs, action, previous_action, previous_observation):
        new_obs = dict()
        for i in range(self.number_of_vehicles):
            new_obs[i] = dict()
            if i == 1:
                new_obs[i]['type'] = 'ego'
            elif obs[i][0] == 1:
                new_obs[i]['type'] = 'sut'
            else:
                new_obs[i]['type'] = 'npc'
            new_obs[i]['x'] = obs[i][2]
            new_obs[i]['y'] = obs[i][3]
            new_obs[i]['v_lon'] = obs[i][4]
            new_obs[i]['v_lat'] = obs[i][5]
            new_obs[i]['a_lon'], new_obs[i]['a_lat'] = self.calculate_acceleration(action[0], obs[i][6], obs[i][7])
            new_obs[i]['j_lon'], new_obs[i]['j_lat'] = self.calculate_jerk(action[0], obs[i][6], obs[i][7], previous_action, previous_observation[i])
        return new_obs

    def step(self, action):
        self.time_step += 1
        new_obs, reward, done, info = super().step(action)
        modified_obs = self.get_modified_obs(new_obs, action, self.previous_action, self.previous_observation)
        self.modifies_observation_log[self.time_step] = modified_obs
        self.previous_action = action
        self.previous_observation = new_obs
        return new_obs, reward, done, info


    def reset(self, **kwargs):
        super().reset()
        self.time_step = 0

    def calculate_acceleration(self, acceleration, cos_h, sin_h):
        acceleration = utils.lmap(acceleration, [-1, 1], self.acceleration_range)
        a_lon = acceleration * cos_h
        a_lat = acceleration * sin_h
        return a_lon, a_lat

    def calculate_jerk(self, acceleration, sin_h, cos_h, previous_action, previous_observation):
        acceleration = utils.lmap(acceleration, [-1, 1], self.acceleration_range)
        pre_acceleration = utils.lmap(previous_action[0], [-1, 1], self.acceleration_range)
        a_lon = acceleration * cos_h
        a_lat = acceleration * sin_h
        pre_a_lon = pre_acceleration * previous_observation[6]
        pre_a_lat = pre_acceleration * previous_observation[7]
        j_lon = (a_lon - pre_a_lon)/self.dt
        j_lat = (a_lat - pre_a_lat)/self.dt
        return j_lon, j_lat


    def get_observation_log(self):
        return self.modifies_observation_log




if __name__ == "__main__":
    # Create the environment
    env_name = "highway-adex-debug-v0"
    reward_name = "purely_adv"
    config = {
        "action": {
            "type": "ContinuousAction"
        },
        "observation": {
            "type": "Kinematics",
            "features": ["is_sut", "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "vehicles_count": 6,
            "absolute": True,
        },
        "vehicles_count": 6,
        "lanes_count": 4
    }

    env = gym.make(env_name)
    env.configure(config)
    obs = env.reset()

    #env = KinematicsExtension(env, env.config["vehicles_count"], env.config["lanes_count"])
    reward_fn = reward_register.make(reward_name, dt=1 / env.config["simulation_frequency"])
    env = HighwayRewardWrapper(env, reward_fn)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering


    for videos in range(10):
        done = False
        obs = env.reset()
        frames = []
        while not done:
            a_t = env.action_space.sample()
            s_t, r_t, done, info = env.step(a_t)
            # print(s_t)
            frame = env.render(mode="rgb_array")
            frames.append(frame)
        writer = imageio.get_writer(f"video_{time.time()}.mp4", fps=5)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    env.close()
