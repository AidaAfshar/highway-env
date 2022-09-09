import numpy as np

from highway_env import utils
from highway_env.envs import HighwayADEXEnv
from highway_env.vehicle.controller import ControlledVehicle
from rewards.reward_wrapper import RewardFunction


class PurelyAdversarialRewardFunction(RewardFunction):
    """
    Simple reward which encourage inducing SUT to crash.
    reward := prize             if ego/sut in collision
    reward := small_penalty     otherwise
    """

    def __init__(self, dt: float):
        self._dt = dt

    def __call__(self, obs, action, next_obs, env: HighwayADEXEnv = None):
        # Compute vehicles' collisions
        ego_vehicle, sut_vehicle = env.vehicle, env.sut_vehicle
        ego_sut_crash, _, _ = ego_vehicle._is_colliding(sut_vehicle, self._dt)

        # Compute the ego-sut crash reward
        reward = -0.01 + 1 * ego_sut_crash
        return reward
