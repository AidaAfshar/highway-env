import math

import numpy as np

from highway_env import utils
from highway_env.envs import HighwayADEXEnv
from highway_env.vehicle.controller import ControlledVehicle
from rewards.reward_wrapper import RewardFunction, HPRSRewardFunction

from typing import List
from typing import Union
import numpy as np
from rewards.reward_functions.hprs import hprs_utils
from rewards.reward_functions.hprs import constants as const

gamma = 1.0
number_of_specs = 5.0


def safety_RSS_potential(state, info):
    assert 'violated_safe_distance' in state
    return 0 if (state['violated_safe_distance'] == 1) else 1


def safety_hard_speed_limit_potential(state, info):
    assert 'violated_hard_speed_limit' in state
    return 0 if (state['violated_hard_speed_limit'] == 1) else 1


def target_reach_destination_potential(state, info):
    assert 'distance_to_target' in state and 'TARGET_DISTANCE' in info
    return 1 - clip_and_norm(state['distance_to_target'], 0, info['TARGET_DISTANCE'])


def comfort_soft_speed_limit_potential(state, info):
    assert 'observation' in state
    assert 'SOFT_SPEED_LIMIT' in info and 'HARD_SPEED_LIMIT' in info
    obs = state['observation']
    assert len(obs) > 0 and len(obs[0]) == 7
    ego_v_lon = obs[0][3]
    return clip_and_norm(info['SOFT_SPEED_LIMIT'] - ego_v_lon, 0, info['HARD_SPEED_LIMIT'])


def comfort_speed_lower_bound_potential(state, info):
    assert 'observation' in state
    assert 'SPEED_LOWER_BOUND' in info and 'HARD_SPEED_LIMIT' in info
    obs = state['observation']
    assert len(obs) > 0 and len(obs[0]) == 7
    ego_v_lon = obs[0][3]
    return clip_and_norm(ego_v_lon - info['SPEED_LOWER_BOUND'], 0, info['HARD_SPEED_LIMIT'])


def comfort_no_faster_than_left_potential(state, info):
    assert 'max_velocity_difference_to_left' in state and 'HARD_SPEED_LIMIT' in info
    return 1 - clip_and_norm(state['max_velocity_difference_to_left'], 0, info['HARD_SPEED_LIMIT'])


def simple_base_reward(state, info):
    assert 'distance_to_target' in state
    assert 'TARGET_DISTANCE' in info and 'TARGET_DISTANCE_TOL' in info
    reached_target = bool(state['distance_to_target'] <= info['TARGET_DISTANCE_TOL'])
    return 1 if reached_target else 0


class HighwayHierarchicalShapingReward(HPRSRewardFunction):

    def __init__(self, dt: float):
        self._dt = dt

    def _safety_potential(self, state, info):
        RSS_reward = safety_RSS_potential(state, info)
        hard_speed_limit_reward = safety_hard_speed_limit_potential(state, info)
        return RSS_reward + hard_speed_limit_reward

    def _target_potential(self, state, info):
        target_reward = target_reach_destination_potential(state, info)
        # hierarchical weights
        safety_weight = safety_RSS_potential(state, info) * safety_hard_speed_limit_potential(state, info)
        return safety_weight * target_reward

    def _comfort_potential(self, state, info):
        c1 = comfort_soft_speed_limit_potential(state, info)
        c2 = comfort_no_faster_than_left_potential(state, info)
        c3 = comfort_speed_lower_bound_potential(state, info)
        comfort_reward = c1 + c2 + c3
        # hierarchical weights
        safety_weight = safety_RSS_potential(state, info) * safety_hard_speed_limit_potential(state, info)
        target_weight = target_reach_destination_potential(state, info)
        return safety_weight * target_weight * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # hierarchical shaping
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = gamma * self._comfort_potential(next_state, info) - self._comfort_potential(state, info)

        '''
        print('time_step: ', info['time_step'])
        print("base_reward: ", base_reward)
        print("shaping_safety : ", shaping_safety)
        print("shaping_target :", shaping_target)
        print("shaping_comfort :", shaping_comfort)
        print("___________________________________")
        '''

        return base_reward + shaping_safety + shaping_target + shaping_comfort


class HighwayScalarizedMultiObjectivization(HPRSRewardFunction):

    def __init__(self, weights: List[float], **kwargs):
        assert len(weights) == number_of_specs, f"nr weights ({len(weights)}) != nr reqs {number_of_specs}"
        assert (sum(weights) - 1.0) <= 0.0001, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_safety = gamma * safety_RSS_potential(next_state, info) - safety_RSS_potential(state, info)
        shaping_target = gamma * target_reach_destination_potential(next_state,
                                                                    info) - target_reach_destination_potential(state,
                                                                                                               info)
        shaping_comfort = gamma * comfort_potential(next_state, info) - comfort_potential(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights,
                        [shaping_safety, shaping_target, shaping_comfort]):
            reward += w * f
        return reward


class HighwayUniformScalarizedMultiObjectivization(HighwayScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(HighwayUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class HighwayDecreasingScalarizedMultiObjectivization(HighwayScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        """
        weights selected considering a budget of 2.0 + 2.0 + 1.0 = 3, then:
            - the sum of safety weights is ~ 2.0/3.0
            - the sum of target weights is ~ 2.0/3.0
            - the sum of comfort weights is ~ 1.0/3.0
        """
        weights = np.array([2.0, 2.0, 1.0])
        weights /= np.sum(weights)
        super(HighwayDecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


def clip_and_norm(v: Union[int, float], minv: Union[int, float], maxv: Union[int, float]):
    """
    utility function which returns the normalized value v' in [0, 1].

    @params: value `v` before normalization,
    @params: `minv`, `maxv` extreme values of the domain.
    """
    return (np.clip(v, minv, maxv) - minv) / (maxv - minv)
