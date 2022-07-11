from abc import ABCMeta, abstractmethod

import gym

from highway_env.envs import AbstractEnv


class RewardFunction:
    @abstractmethod
    def __call__(self, obs, action, next_obs, env: AbstractEnv = None):
        pass

    @abstractmethod
    def done(self, obs, done=False, env: AbstractEnv = None):
        return done


class HighwayRewardWrapper(gym.Wrapper, AbstractEnv):
    def __init__(self, env, reward_fn: RewardFunction):
        super(HighwayRewardWrapper, self).__init__(env)
        self._reward_fn = reward_fn
        self._obs = None

    def reset(self, **kwargs):
        self._obs = super(HighwayRewardWrapper, self).reset(**kwargs)
        return self._obs

    def step(self, action):
        next_obs, base_reward, done, info = super(HighwayRewardWrapper, self).step(action)
        custom_reward = self._reward_fn(self._obs, action, next_obs, env=self)
        done = self._reward_fn.done(next_obs, done, env=self)
        self._obs = next_obs
        return self._obs, custom_reward, done, info
