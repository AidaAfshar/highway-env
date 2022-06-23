import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class HighwayADEXEnv(AbstractEnv):
    """
    A highway driving environment extended to handle EGO and SUT vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "features": ["is_sut", "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 2,  # we have 2 controlled vehicles: the 'ego' and the 'sut'
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @property
    def sut_vehicle(self) -> Vehicle:
        """Last (default) controlled vehicle."""
        if len(self.controlled_vehicles) <= 1:
            msg = f"try to use ego-sut vehicles but only {len(self.controlled_vehicles)} controller vehicles."
            raise EnvironmentError(msg)
        return self.controlled_vehicles[-1] if self.controlled_vehicles else None

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for i, others in enumerate(other_per_controlled):
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            # create controlled vehicle
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            # choose role of controlled vehicle
            role = 'sut' if i == len(other_per_controlled) - 1 else 'ego'   # only 1 'sut' (last)
            vehicle.role = role

            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """

        # compute reward for ego and sut vehicles
        rewards_per_role = {'ego': [], 'sut': []}
        for vehicle in self.road.vehicles:
            if vehicle.role == 'npc':
                continue
            neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
            lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
                else vehicle.lane_index[2]
            # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
            forward_speed = vehicle.speed * np.cos(vehicle.heading)
            scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
            reward = \
                + self.config["collision_reward"] * vehicle.crashed \
                + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
                + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
            reward = utils.lmap(reward,
                              [self.config["collision_reward"],
                               self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                              [0, 1])
            reward = 0 if not vehicle.on_road else reward
            # store (role, reward)
            rewards_per_role[vehicle.role] += [reward]
        # aggregate rewards
        ego_reward = np.mean(rewards_per_role['ego'])   # in case controlling >1 agent
        sut_reward = rewards_per_role['sut'][0]         # the 'sut' in unique
        reward = utils.lmap(ego_reward - sut_reward,    # Rescale reward to be in [0, 1]
                            [-1, 1],                    # min reward: ego_reward=0, sut_reward=1 -> -1
                            [0, 1])                     # max_reward: ego_reward=1, sut_reward=0 -> +1
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayADEXEnvFast(HighwayADEXEnv):
    """
    A variant of highway-adex-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class HighwayADEXEnvDebug(HighwayADEXEnvFast):
    """
    A variant of highway-adex-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 1,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg




register(
    id='highway-adex-v0',
    entry_point='highway_env.envs:HighwayADEXEnv',
)

register(
    id='highway-adex-fast-v0',
    entry_point='highway_env.envs:HighwayADEXEnvFast',
)

register(
    id='highway-adex-debug-v0',
    entry_point='highway_env.envs:HighwayADEXEnvDebug',
)
