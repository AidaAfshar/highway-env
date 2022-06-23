import functools
from typing import Tuple, Dict, Optional

import gym
from pettingzoo.utils import random_demo, parallel_to_aec

import highway_env
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionDict, ObsDict, AgentID


class AdversarialHighwayEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "adv_highway_v1"}

    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make('highway-v0')
        self.env.seed(0)
        self.possible_agents = [0, 1]


        self.env.configure({"controlled_vehicles": 3})  # Two controlled vehicles
        self.env.configure({"vehicles_count": 0})  # A single other vehicle, for the sake of visualisation
        self.env.configure({
            "ego_spacing": 2,
            "controlled_vehicles": 2,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                }
            }
        })
        self.env.reset()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gym.Space:
        return self.env.observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gym.Space:
        return self.env.action_space[agent]

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:
        obs = self.env.reset()
        self.agents = self.possible_agents.copy()
        return self.to_dict(obs)

    def seed(self, seed=None):
        pass

    def step(self, actions: ActionDict) -> Tuple[ObsDict, Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        if len(actions) == 0:
            return None
        actions = tuple([actions[i] for i in sorted(actions)])
        obs, rewards, done, info = self.env.step(actions)
        step = self.to_dict(obs), {i:  rewards if i == 0 else -rewards for i in self.agents}, {i: done for i in self.agents}, {i: info for i in self.agents}
        if done:
            self.agents.clear()

        return step

    def render(self, mode="human"):
        return self.env.render(mode)

    def state(self) -> np.ndarray:
        pass

    def to_dict(self, list):
        return {i: item for i, item in enumerate(list)}

if __name__ == '__main__':
    import su
    from pettingzoo.test import parallel_api_test
    env = AdversarialHighwayEnv()

    env.reset()
    #parallel_api_test(env, num_cycles=1000)
    #random_demo(parallel_to_aec(env), render=True, episodes=5)
