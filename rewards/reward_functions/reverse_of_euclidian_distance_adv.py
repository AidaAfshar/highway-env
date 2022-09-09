import numpy as np

from highway_env import utils
from highway_env.envs import HighwayADEXEnv
from highway_env.vehicle.controller import ControlledVehicle
from rewards.reward_wrapper import RewardFunction

class ReverseOfRelativeDistanceRewardFunction(RewardFunction):
    """
    Simple reward which encourage ego get close (and probably crash) with SUT.
    reward := 1 / 1+(euclidian distance of sut and ego)
    """

    def __init__(self, dt: float):
        self._dt = dt

    def __call__(self, obs, action, next_obs, env: HighwayADEXEnv = None):
        ego_vehicle, sut_vehicle = env.vehicle, env.sut_vehicle
        dist_square = self.get_euclidian_disctance(ego_vehicle, sut_vehicle)
        reward = 1/(1+dist_square)
        return reward

    def get_number_of_features(self, env):
        return len(env.config["observation"]["features"])

    def read_observation(self, obs, number_of_features):
        observations = []
        for i in range(number_of_features):
            vehicle_obs = obs[i * number_of_features:(i + 1) * number_of_features:]
            observations.append(vehicle_obs)
        return observations

    def get_ego_observation(self, obs, env):
        number_of_features = self.get_number_of_features(env)
        observations = self.read_observation(obs, number_of_features)
        return observations[0]

    def get_sut_relative_position(self, obs, env):
        observations = self.get_ego_observation(obs, env)
        for i in range(len(observations)):
            if observations[i][0]:
                dx = observations[i][2]
                dy = observations[i][3]
                print(dx, "  -  ", dy)
                return dx, dy
        return None, None


    def get_euclidian_disctance(self, ego_vehicle, sut_vehicle):
        ego_position = ego_vehicle.position
        sut_position = sut_vehicle.position
        ego_x = ego_position[0]
        ego_y = ego_position[1]
        sut_x = sut_position[0]
        sut_y = sut_position[1]
        dx = ego_x - sut_x
        dy = ego_y - sut_y
        dist_square = pow(dx, 2) + pow(dy, 2)
        return dist_square