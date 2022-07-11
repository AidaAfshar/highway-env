import numpy as np

from highway_env import utils
from highway_env.envs import HighwayADEXEnv, AbstractEnv
from highway_env.vehicle.controller import ControlledVehicle
from rewards.reward_wrapper import RewardFunction
from training.utils import clip_and_norm

gamma = 1.0


class PurelyAdversarialRewardFunction(RewardFunction):
    """
    Simple reward which encourage inducing SUT to crash.
    reward := prize             if ego/sut in collision
    reward := small_penalty     otherwise
    """

    def __init__(self, env):
        self._dt = 1 / env.config["simulation_frequency"]

    def __call__(self, obs, action, next_obs, env: HighwayADEXEnv = None):
        # Compute vehicles' collisions
        ego_vehicle, sut_vehicle = env.vehicle, env.sut_vehicle
        ego_sut_crash, _, _ = ego_vehicle._is_colliding(sut_vehicle, self._dt)

        # Compute the ego-sut crash reward
        reward = -0.01 + 1 * ego_sut_crash
        return reward


class BreakLongitudinalAndLateralDistance(RewardFunction):
    """
    Informal description:
    - the sut aims to avoid collision
        sqrt d_lon**2 + d_lat **2 >= d_collision

    - the ado aims to break the sut spec while performing a cut-in maneuver:
        achieve (d_lon >= vehicle_len and d_lat < 1)
        ensure (no collision with other vehicles)
        ensure (d_lon <= max_dist)
    """

    def __init__(self, env):
        self._vehicle_len = 1.0  # educated guess
        self._d_collision = 0.5  # educated guess
        self._max_dist = 5.0

    def _compute_base_reward(self, obs, action, next_obs, env):
        """ achieve (d_lon >= vehicle_len and |d_lat| <= 1) """
        ego_vehicle, sut_vehicle = env.vehicle, env.sut_vehicle
        d_lon = (ego_vehicle.position[0] - sut_vehicle.position[0])
        d_lat = (ego_vehicle.position[1] - sut_vehicle.position[1])
        if d_lon >= self._vehicle_len and abs(d_lat) < self._d_collision:
            return 1.0
        return 0.0

    def _compute_target_potential(self, obs):
        """ achieve (d_lon >= vehicle_len and |d_lat| <= 1) """
        safety_potential = self._compute_safety_potential(obs)
        d_lon, d_lat = obs["d_lon"][0], obs["d_lat"][0]
        pred_1 = clip_and_norm(d_lon, -10, self._vehicle_len)  # as in hprs
        pred_2 = clip_and_norm(d_lat, self._d_collision, +2)  # as in hprs
        return safety_potential * (0.5 * (pred_1 + pred_2))

    def _compute_safety_potential(self, obs):
        safety_reqs = obs["nonsut_collision"] <= 0 and obs["d_lon"] <= self._max_dist
        return 1.0 if safety_reqs else 0.0

    def __call__(self, obs, action, next_obs, env: AbstractEnv = None):
        reward = self._compute_base_reward(obs, action, next_obs, env)
        # shaping
        safety_shaping = gamma * self._compute_safety_potential(next_obs) - self._compute_safety_potential(obs)
        target_shaping = gamma * self._compute_target_potential(next_obs) - self._compute_target_potential(obs)
        return reward + safety_shaping + target_shaping

    def done(self, obs, done=False, env: AbstractEnv = None):
        safety_score = self._compute_safety_potential(obs)
        return done or (safety_score <= 0)


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
