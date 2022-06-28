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


class DrivingRewardPlusAdversarial(RewardFunction):
    """
    Simple reward which encourage inducing SUT to crash.
    There are two terms:
        1. reward ego driving
        2. reward ego-sut crash
    Note that without the first term, the ego does not show a decent,
    without the second term, the ego is not incentivate in crashing the sut.
    """

    def __init__(self, dt: float):
        self._dt = dt

    def __call__(self, obs, action, next_obs, env: HighwayADEXEnv = None):
        # Compute vehicles' collisions
        ego_vehicle, sut_vehicle = env.vehicle, env.sut_vehicle
        ego_crashed, sut_crashed = ego_vehicle.crashed, sut_vehicle.crashed
        ego_sut_crash, _, _ = ego_vehicle._is_colliding(sut_vehicle, self._dt)

        # Compute reward to incentivate good driving behavior
        # The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        neighbours = env.road.network.all_side_lanes(ego_vehicle.lane_index)
        lane = ego_vehicle.target_lane_index[2] if isinstance(ego_vehicle, ControlledVehicle) else \
        ego_vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = ego_vehicle.speed * np.cos(ego_vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, env.config["reward_speed_range"], [0, 1])
        ego_reward = env.config["collision_reward"] * (ego_crashed and not ego_sut_crash) \
                 + env.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
                 + env.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        ego_reward = utils.lmap(ego_reward,
                            [env.config["collision_reward"],
                             env.config["high_speed_reward"] + env.config["right_lane_reward"]],
                            [0, 1])
        ego_reward = 0 if not ego_vehicle.on_road else ego_reward

        # Compute the ego-sut crash reward
        ego_sut_reward = 1 * ego_sut_crash

        # Aggregate in final reward
        reward = utils.lmap(ego_reward + ego_sut_reward, [0, 2], [0, 1])
        return reward
