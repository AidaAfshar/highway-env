import gym
from highway_env.envs import AbstractEnv, HighwayEnvHPRS

class FilterObservationWrapper(gym.Wrapper):
    """
    observation wrapper that filter a single observation and return is without dictionary,
    all the observable quantities are moved to the info as `state`
    """

    def __init__(self, env, obs_list=[]):
        super(FilterObservationWrapper, self).__init__(env)
        self._obs_list = obs_list
        self.observation_space = gym.spaces.Dict({obs: self.env.observation_space[obs] for obs in obs_list})

    def _filter_obs(self, original_obs):
        new_obs = {}
        for obs in self._obs_list:
            assert obs in original_obs
            new_obs[obs] = original_obs[obs]
        return new_obs

    def step(self, action):
        original_obs, reward, done, info = super().step(action)
        new_obs = self._filter_obs(original_obs)
        # add original state into the info
        new_info = info
        new_info['state'] = {name: value for name, value in original_obs.items()}
        return new_obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        new_obs = self._filter_obs(obs)
        return new_obs


class HPRSFilterObservationWrapper(gym.Wrapper):
    """
    observation wrapper that filter a single observation and return is without dictionary,
    all the observable quantities are moved to the info as `state`
    """

    def __init__(self, env, obs_list=[]):
        super(HPRSFilterObservationWrapper, self).__init__(env)
        self._obs_list = obs_list
        self.observation_space = self.env.observation_space

    def _filter_obs(self, original_obs):
        assert 'observation' in self._obs_list
        new_obs = original_obs['observation']
        return new_obs

    def step(self, action):
        original_obs, reward, done, info = super().step(action)
        new_obs = self._filter_obs(original_obs)
        # add original state into the info
        new_info = info
        new_info['state'] = {name: value for name, value in original_obs.items()}
        return new_obs, reward, done, new_info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        new_obs = self._filter_obs(obs)
        return new_obs
