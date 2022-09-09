from abc import ABC, ABCMeta, abstractmethod

import gym

from highway_env.envs import AbstractEnv, HighwayEnvHPRS


class RewardFunction:
    @abstractmethod
    def __call__(self, obs, action, next_obs, env: AbstractEnv = None):
        pass

    @abstractmethod
    def done(self, obs, done=False, env: AbstractEnv = None):
        return done


class HPRSRewardFunction(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        pass


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


class HPRSHighwayRewardWrapper(gym.Wrapper, HighwayEnvHPRS):

    def __init__(self, env, reward_fn: HPRSRewardFunction):
        super(HPRSHighwayRewardWrapper, self).__init__(env)
        self._reward_fn = reward_fn
        self._state = None
        self._reward = 0.0
        self._return = 0.0

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self._state = state
        self._reward = 0.0
        self._return = 0.0
        return self._state

    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        reward = self._reward_fn(state=self._state, action=action, next_state=next_state, info=info)
        self._state = next_state
        self._reward = reward
        self._return += reward
        return next_state, reward, done, info

