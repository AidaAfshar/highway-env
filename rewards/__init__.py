from typing import Callable, Any

from rewards.reward_fns import PurelyAdversarialRewardFunction, ReverseOfRelativeDistanceRewardFunction
from rewards.reward_functions.hprs.potential import HighwayHierarchicalShapingReward
from rewards.reward_wrapper import RewardFunction, HPRSRewardFunction


class RewardRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, reward_name: str, reward_factory: Callable[[Any], RewardFunction]):
        assert reward_name not in self._registry, f"reward {reward_name} already defined in RewardRegister"
        self._registry[reward_name] = reward_factory

    def make(self, reward_name, **kwargs):
        assert reward_name in self._registry, f"reward {reward_name} not defined in RewardRegister"
        return self._registry[reward_name](**kwargs)


class HPRSRewardRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, reward_name: str, reward_factory: Callable[[Any], HPRSRewardFunction]):
        assert reward_name not in self._registry, f"reward {reward_name} already defined in RewardRegister"
        self._registry[reward_name] = reward_factory

    def make(self, reward_name, **kwargs):
        assert reward_name in self._registry, f"reward {reward_name} not defined in RewardRegister"
        return self._registry[reward_name](**kwargs)


reward_register = RewardRegistry()
reward_register.register("purely_adv", PurelyAdversarialRewardFunction)
reward_register.register("reverse_of_euclidian_distance_adv", ReverseOfRelativeDistanceRewardFunction)

hprs_reward_register = HPRSRewardRegistry()
hprs_reward_register.register("hprs", HighwayHierarchicalShapingReward)



